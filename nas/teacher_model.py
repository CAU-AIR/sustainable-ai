import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from ofa.utils.run_config import FaceRunConfig
from ofa.utils.face_data import PairFaceDataset
import models.networks.resnets as resnet
from ofa.utils.common_tools import DistributedMetric


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
    
        train_accuracy = 100 * correct / total
        epoch_loss = running_loss / total_batches
        print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}/{total_batches} - Loss: {epoch_loss:.4f}, Training accuracy: {train_accuracy:.2f}%')


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

def update_face_metric(self, metric_dict, feats):
    FPRs=['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']
    dim = feats.shape[-1]

    # pair-wise scores
    feats = F.normalize(feats.reshape(-1, dim), dim=1)
    feats = feats.reshape(-1, 2, dim)
    feats0 = feats[:, 0, :]
    feats1 = feats[:, 1, :]
    scores = torch.sum(feats0 * feats1, dim=1).tolist()

    retrieval_targets = self.run_config.data_provider.test.dataset.retrieval_targets        

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

def get_metric_vals(self, metric_dict, return_dict=False):
    if return_dict:
        return {key: metric_dict[key].avg.item() for key in metric_dict}
    else:
        return [metric_dict[key].avg.item() for key in metric_dict]

def test(args, net, testloader, device):
    net.eval()
    
    mb_size = args.test_batch_size
    n_samples = args.test_size
    output_dim = max(net.input_channel) * 32
    feats = torch.zeros([n_samples, 2, output_dim], dtype=torch.float32).to(device)
    metric_dict = get_metric_dict()

    with torch.no_grad():
        for idx, data in  enumerate(testloader):
            query_x, retrieval_x, labels = data
            query_x, retrieval_x, labels = query_x.to(device), retrieval_x.to(device), labels.to(device)
            
            # compute output
            qeury_feat = net(query_x, return_feature=True)
            retrieval_feat = net(retrieval_x, return_feature=True)

            batch_start_idx = idx * mb_size
            actual_batch_size = qeury_feat.size(0)
            batch_end_idx = batch_start_idx + actual_batch_size

            feats[batch_start_idx:batch_end_idx, 0, :] = qeury_feat
            feats[batch_start_idx:batch_end_idx, 1, :] = retrieval_feat

            # measure accuracy
            update_face_metric(metric_dict, feats.cpu())

    return get_metric_vals(metric_dict)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # train_dataset = CASIAWebFaceDataset(all_files, transform=transform)
    train_provider = FaceRunConfig()
    train_dataset = train_provider.data_provider.train_dataset(transform)

    test_path = 'dataset/face/test_lfw/'
    test_dataset = PairFaceDataset(root=test_path, 
                                   transform=transform, 
                                   data_annot=test_path)
    args.test_size = (len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    n_classes = train_provider.data_provider.n_classes
    teacher_model = resnet.ResNet50(n_classes)
    teacher_model = nn.DataParallel(teacher_model)
    teacher_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)

    best_accuracy = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        print('epoch : ', epoch)
        train(epoch, teacher_model, train_loader, optimizer, device)
        # val_accuracy = validate(teacher_model, valid_loader, device)
        test_accuracy, _ = test(args, teacher_model, test_loader, device)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(teacher_model.state_dict(), 'best_model.pth.tar')

    print(f'Saved Best Model with Test Accuracy: {best_accuracy:.2f}%')

