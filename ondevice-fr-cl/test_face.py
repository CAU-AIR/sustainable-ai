from sklearn import metrics # it needs to be imported first in Jetson Nano
import argparse
import os
import pickle
import psutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets import *
from scenario import *
from models import *
from method import *
from scenario.evaluators import *

from utils import seedEverything, create_if_not_exists, AverageMeter

import wandb


def parse_args():
    parser = argparse.ArgumentParser('argument')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--method', type=str, default='lwf')
    parser.add_argument('--target_device', type=str, default='pc', choices=['pc', 'nano'])

    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=5)

    parser.add_argument('--save_path', type=str, default='save/casia_15')
    parser.add_argument('--save_model', type=int, default=1, choices=[0, 1])

    parser.add_argument('--nowand', type=int, default=0, choices=[0, 1])
    parser.add_argument('--debug_mode', type=int, default=0, choices=[0, 1])

    parser.add_argument('--wandb_project', type=str, default='nano')
    parser.add_argument('--wandb_entity', type=str, default='laymond1')

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def main(args):
    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()
        
    # seed
    seedEverything(args.seed)
    
    CASIAWEBFACE = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])
    train_transform = CASIAWEBFACE
    # Train & Test dataset and scenario
    casia_dataset = CASIAWeb15Dataset(root="./data/CASIA-15/", 
                        transform=train_transform)
    lfw_test_dataset = LFWPairDataset(root="./data/",
                        transform=None,
                        data_annot="./data/")
    
    # scenario
    train_scenario = ClassIncremental(
        dataset=casia_dataset, n_tasks=casia_dataset._DEFAULT_N_TASKS, batch_size=args.batch_size, n_workers=args.n_workers
    )

    # Verification Scenario: n_tasks arguments 필요 없음.
    # LFW Test Dataset
    lfw_test_scenario = VerificationScenario(
        dataset=lfw_test_dataset, n_tasks=lfw_test_dataset._DEFAULT_N_TASKS, batch_size=args.test_batch_size, n_workers=args.n_workers
    )   
    
    # 
    n_classes = casia_dataset.n_classes()
    args.N_CLASSES_PER_TASK = casia_dataset._N_CLASSES_PER_TASK
    args._DEFAULT_N_TASKS = casia_dataset._DEFAULT_N_TASKS
    
    # model
    net = resnet18(n_classes)
    loss = torch.nn.CrossEntropyLoss()
    method = LwF(net, loss, args, None)    

    # evaluator
    lfw_rep_evaluator = VerificationEvaluator(method=method, eval_scenario=lfw_test_scenario, name="Verification")
    test_evaluators = [lfw_rep_evaluator]


    # TODO: logger (wandb & local & tensorboard)
    # def set_loggers()
    #     raise NotImplementedError
    
    # save path
    if args.save_path is not None:
        args.save_path = os.path.join(args.save_path, method.NAME)
        create_if_not_exists(args.save_path)
    
    print(f"Number of tasks: {train_scenario.n_tasks} | Number of classes: {train_scenario.n_classes}")
    
    metrics = []
    # train
    method.train()
    method.net.to(method.device)
    for task_id, train_loader in enumerate(train_scenario):
        #- Start Epoch
        scheduler = None
        # pbar = tqdm(range(args.n_epochs))
        for epoch in range(args.n_epochs):
            #-- Start Iteration
            batch_time = AverageMeter()
            losses = AverageMeter()
            
            pbar = tqdm(train_loader)
            end = time.time()
            for idx, (inputs, labels, task, not_aug_inputs) in enumerate(pbar):
                if args.debug_mode and idx > 5:
                    break
                inputs, labels, not_aug_inputs = inputs.to(method.device), labels.to(method.device), not_aug_inputs.to(method.device)

                if hasattr(method, 'meta_observe'):
                    loss = method.meta_observe(inputs, labels, not_aug_inputs)
                else:
                    loss = method.observe(inputs, labels, not_aug_inputs)
                losses.update(loss, inputs.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                pbar.set_description("[Task:{}|Epoch:{}] Avg Loss: {:.5}".format(task_id+1, epoch+1, losses.avg))
                wandb.log({'Batch Time': batch_time.avg})
                # Memory Usage in MB
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 ** 2)
                wandb.log({'Memory Usage(MB) in iter': memory_mb})
            
            if scheduler is not None:
                scheduler.step()
            
            # pbar.set_description(f"[Task:{task_id+1}|Epoch:{epoch+1}] Avg Loss: {losses.avg:.5}")
            # pbar.set_description("[Task:{}|Epoch:{}] Avg Loss: {:.5}".format(task_id+1, epoch+1, losses.avg))

        
        #- Start Evaluation
        accs = test_evaluators[0].fit(current_task_id=task_id, logger=None)
        metrics.append(accs)
        wandb.log({'Task-Verif ACC': accs})
        # wandb.log('Task-{task_id}-Verif ACC: {acc}'.format(task_id=task_id, acc=accs), step=epoch)
    
        if args.save_path is not None:
            if args.save_model:
                fname = os.path.join(args.save_path, "{}_{}.pth".format(method.NAME, task_id+1))
                torch.save(method.net.state_dict(), fname)
    
    if args.save_path is not None:           
        # save the metrics
        fname = os.path.join(args.save_path, "{}.pkl".format(method.NAME))
        with open(file=fname, mode='wb') as f:
            pickle.dump({'ACC': np.array(metrics)}, f)
                
    
if __name__ == "__main__":
    # class Config:
    #     target_device = 'pc'
    #     seed = 0
    #     n_workers = 4
    #     batch_size = 512
    #     lr = 0.01
    #     nowand = 0
    #     n_epochs = 20
    #     debug = 0
    #     save_path = 'save/casia_15'
    #     save_model = 1
    #     wandb_project ='nano'
    #     wandb_entity ='laymond1'
    # args = Config

    args = parse_args()
    
    main(args)