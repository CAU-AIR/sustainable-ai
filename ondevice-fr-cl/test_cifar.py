from sklearn import metrics # it needs to be imported first in Jetson Nano
import argparse
from copy import deepcopy
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
    parser.add_argument('--datasets', type=str, default='cifar10')

    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=1)
    
    # LwF Args
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=2)
    parser.add_argument('--wd_reg', type=float, default=1.0)

    parser.add_argument('--save_path', type=str, default='save/cifar10')
    parser.add_argument('--save_model', type=int, default=1, choices=[0, 1])

    parser.add_argument('--relogin', type=int, default=0, choices=[0, 1])
    parser.add_argument('--nowand', type=int, default=0, choices=[0, 1])
    parser.add_argument('--debug_mode', type=int, default=0, choices=[0, 1])

    parser.add_argument('--wandb_project', type=str, default='nano')
    parser.add_argument('--wandb_entity', type=str, default='laymond1')

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def main(args):
    if not args.nowand:
        wandb.login(relogin=args.relogin)
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()
        
    # seed
    seedEverything(args.seed)
    
    TRANSFORM10 = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                        (0.2470, 0.2435, 0.2615))])
    test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.2470, 0.2435, 0.2615))])
    train_transform = TRANSFORM10
    
    # scenario
    splitcifar10 = SplitCIFAR10(root='./data/', transform=train_transform)
    paircifar10 = PairCIFAR10(root='./data/', transform=test_transform, 
                              data_annot='./data/')
    train_scenario = ClassIncremental(
            dataset=splitcifar10, n_tasks=splitcifar10._DEFAULT_N_TASKS, batch_size=args.batch_size, n_workers=args.n_workers
        )
    test_scenario = VerificationScenario(
        dataset=paircifar10, n_tasks=paircifar10._DEFAULT_N_TASKS, batch_size=args.test_batch_size, n_workers=args.n_workers
        )

    # 
    n_classes = splitcifar10.n_classes()
    args.N_CLASSES_PER_TASK = splitcifar10._N_CLASSES_PER_TASK
    args._DEFAULT_N_TASKS = splitcifar10._DEFAULT_N_TASKS
    
    # model
    net = resnet18(n_classes)
    loss = torch.nn.CrossEntropyLoss()
    method = LwF(net, loss, args, None)    

    # evaluator
    cifar10_verif_evaluator = VerificationEvaluator(method=method, eval_scenario=test_scenario, name="Verification")

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
        # begin task
        if hasattr(method, 'begin_task'):
            method.begin_task(train_loader)
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
        accs = cifar10_verif_evaluator.fit(current_task_id=1, logger=None)
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
    #     seed = 0
    #     batch_size = 16
    #     n_workers = 0
    #     lr = 0.01
    #     nowand = 1
    #     n_epochs = 20
    #     debug = 0
    #     save_path = 'save/cifar10'
    #     save_model = 1
    #     wandb_project ='nano'
    #     wandb_entity ='laymond1'
    # args = Config

    args = parse_args()
    
    main(args)