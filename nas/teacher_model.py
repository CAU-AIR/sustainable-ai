import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from ofa.utils.face_data import PairFaceDataset, FaaceDataProvider
import models.networks.common_resnet as resnet
# import models.networks.resnets as resnet
from ofa.utils.common_tools import DistributedMetric

import wandb


def train(epoch, net, trainloader, optimizer, device):
    net.train()
    total = 0
    correct = 0
    running_loss = 0.0
    total_batches = len(trainloader)

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        batch_loss = loss.item()
        batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
        print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}/{total_batches} - Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.2f}%')

    
    epoch_loss = running_loss / total_batches
    epoch_accuracy = 100 * correct / total

    print(f'Epoch {epoch + 1} - Average Loss: {epoch_loss:.4f}, Average Training Accuracy: {epoch_accuracy:.2f}%')

    return epoch_loss, epoch_accuracy


def face_accuracy(labels, scores, FPRs):
    from sklearn import metrics
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    # eer and auc
    # len(labels) = 6000
    # scores shape = 1048576 -> must be 6,000
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100. * brentq(lambda x : 1. - x - roc_curve(x), 0., 1.)
    AUC = 100. * metrics.auc(fpr, tpr)

    # get acc
    tnr = 1. - fpr
    # pos_num = labels.count(1)
    pos_num = len(labels.nonzero()[0])
    neg_num = len(labels) - pos_num
    ACC = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)

    # TPR @ FPR
    if isinstance(FPRs, list):
        TPRs = [
            ('TPR@FPR={}'.format(FPR), 100. * roc_curve(float(FPR)))
            for FPR in FPRs
        ]
    else:
        TPRs = []

    return [('ACC', ACC), ('EER', EER), ('AUC', AUC)] + TPRs

def update_face_metric(metric_dict, feats, dataset):
    FPRs=['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']
    dim = feats.shape[-1]

    # pair-wise scores
    feats = F.normalize(feats.reshape(-1, dim), dim=1)
    feats = feats.reshape(-1, 2, dim)
    feats0 = feats[:, 0, :]
    feats1 = feats[:, 1, :]
    scores = torch.sum(feats0 * feats1, dim=1).tolist()

    retrieval_targets = dataset.retrieval_targets

    results = face_accuracy(retrieval_targets, scores, FPRs)
    results = dict(results)
    metric = ['ACC']
    # metric_dict["top1"].update(results[metric[0]], scores.szie(0))
    metric_dict["top1"].update(results[metric[0]], len(scores))

def get_metric_dict():
    return {
        "top1": DistributedMetric("top1"),
        "top5": DistributedMetric("top5"),
    }

def get_metric_vals(metric_dict, return_dict=False):
    if return_dict:
        return {key: metric_dict[key].avg.item() for key in metric_dict}
    else:
        return [metric_dict[key].avg.item() for key in metric_dict]

def test(args, net, testloader, device, test_dataset):
    net.eval()
    test_latency = 0
    total_samples = 0
    
    mb_size = args.batch
    n_samples = args.test_size
    # output_dim = max(net.input_channel) * 32
    output_dim = net.module.feature_size # 2048
    feats = torch.zeros([n_samples, 2, output_dim], dtype=torch.float32).to(device)
    metric_dict = get_metric_dict()

    with torch.no_grad():
        for idx, data in  enumerate(testloader):
            start_time = time.time()

            query_x, retrieval_x, labels = data
            query_x, retrieval_x, labels = query_x.to(device), retrieval_x.to(device), labels.to(device)
            
            # compute output
            # qeury_feat = net(query_x, outputs='features')
            # retrieval_feat = net(retrieval_x, outputs='features')
            qeury_feat = net.features(query_x)
            retrieval_feat = net.features(retrieval_x)

            batch_start_idx = idx * mb_size
            actual_batch_size = qeury_feat.size(0)
            batch_end_idx = batch_start_idx + actual_batch_size

            feats[batch_start_idx:batch_end_idx, 0, :] = qeury_feat
            feats[batch_start_idx:batch_end_idx, 1, :] = retrieval_feat

            # measure accuracy
            update_face_metric(metric_dict, feats.cpu(), test_dataset)

            test_latency += time.time() - start_time
            total_samples += len(data[0])

        avg_latency = test_latency / total_samples

    return get_metric_vals(metric_dict), avg_latency

def calculate_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--model", type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'])
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # To solve RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_provider = FaaceDataProvider(save_path="/home/ml/sustainable-ai/nas/dataset")
    train_dataset = train_provider.train_dataset(transform)

    test_path = '/home/ml/sustainable-ai/nas/dataset/test_lfw/'
    test_dataset = PairFaceDataset(root=test_path, 
                                   transform=transform, 
                                   data_annot=test_path)
    args.test_size = (len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    n_classes = train_provider.n_classes
    teacher_model, model_name = resnet.__dict__[args.model](n_classes)
    teacher_model = nn.DataParallel(teacher_model)
    teacher_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 50
    best_accuracy = 0.
    total_latency = 0.

    model_size = calculate_model_size(teacher_model)
    
    args.save_path = "exp/" + args.model
    os.makedirs(args.path, exist_ok=True)

    logs = wandb
    login_key = '1623b52d57b487ee9678660beb03f2f698fcbeb0'
    logs.login(key=login_key)
    # wandb.init(config=args, project='ResNet for OFA Dist', name=model_name)
    wandb.init(config=args, project='ResNet for OFA Dist', name='ResNet50')

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(epoch, teacher_model, train_loader, optimizer, device)
        # val_accuracy = validate(teacher_model, valid_loader, device)
        test_accuracy, avg_latency = test(args, teacher_model, test_loader, device, test_dataset)

        total_latency += avg_latency

        logs.log({"Train Loss": train_loss})
        logs.log({"Train Acc": train_accuracy})
        logs.log({"Test Acc": test_accuracy[0]})


        if test_accuracy[0] > best_accuracy:
            best_accuracy = test_accuracy[0]
            torch.save(teacher_model.state_dict(), args.save_path + '/best_model.pth.tar')

    epoch_latency = total_latency / num_epochs
    logs.log({"Model Size": model_size, "Latency": epoch_latency})

    print(f'Saved Best Model with Test Accuracy: {best_accuracy:.2f}%')