from sklearn import metrics # it needs to be imported first in Jetson Nano
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import *
from scenario import *
from models import *
from method import *
from scenario.evaluators import *

from utils import seedEverything, create_if_not_exists, AverageMeter


def main(args):
    # seed
    seedEverything(0)
    
    # scenario
    splitcifar10 = SplitCIFAR10(root='./data')
    paircifar10 = PairCIFAR10(root='./data')
    train_scenario = ClassIncremental(
            dataset=splitcifar10, n_tasks=splitcifar10._DEFAULT_N_TASKS, batch_size=args.batch_size, n_workers=0
        )
    test_scenario = VerificationScenario(
        dataset=paircifar10, n_tasks=paircifar10._DEFAULT_N_TASKS, batch_size=args.batch_size, n_workers=0
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
    
    metrics = []
    # train
    method.train()
    method.net.to(method.device)
    for task_id, train_loader in enumerate(train_scenario):
        #- Start Epoch
        scheduler = None
        pbar = tqdm(range(args.n_epochs))
        for epoch in pbar:
            #-- Start Iteration
            losses = AverageMeter()
            for idx, (inputs, labels, task, not_aug_inputs) in enumerate(train_loader):
                if args.debug and idx > 5:
                    break
                inputs, labels, not_aug_inputs = inputs.to(method.device), labels.to(method.device), not_aug_inputs.to(method.device)

                if hasattr(method, 'meta_observe'):
                    loss = method.meta_observe(inputs, labels, not_aug_inputs)
                else:
                    loss = method.observe(inputs, labels, not_aug_inputs)
                losses.update(loss, inputs.size(0))
            
            if scheduler is not None:
                scheduler.step()
            
            # pbar.set_description(f"[Task:{task_id+1}|Epoch:{epoch+1}] Avg Loss: {losses.avg:.5}")
            pbar.set_description("[Task:{}|Epoch:{}] Avg Loss: {:.5}".format(task_id+1, epoch+1, losses.avg))

        
        #- Start Evaluation
        accs = cifar10_verif_evaluator.fit(current_task_id=1, logger=None)
        metrics.append(accs)
    
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
    class Config:
        batch_size = 8
        lr = 0.01
        nowand = 1
        n_epochs = 3
        debug = 1
        save_path = 'save/cifar10'
        save_model = 1

    args = Config
    main(args)