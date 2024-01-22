import random
import argparse
import numpy as np
import horovod.torch as hvd

import torch
import torch.nn as nn
import torchvision.models as models

from ofa.utils.run_config import DistributedImageNetRunConfig

def train(epoch, net, trainloader, optimizer, device):
    net.train()
    total = 0
    correct = 0
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

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
    epoch_loss = running_loss / len(trainloader)
    print(f'Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Training accuracy: {train_accuracy:.2f}%')
        

def validate(net, valloader, device):
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    return val_accuracy

def test(net, testloader, device):
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    return test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "tinyimagenet", "casiaweb"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    hvd.init()
    num_gpus = hvd.size()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")

    expansion = 4

    teacher_model = models.resnet101()
    teacher_model.fc = nn.Linear(512*expansion, 200)
    teacher_model.to(device)

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # load tinyImageNet
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())
    
    train_loader = run_config.train_loader
    valid_loader = run_config.valid_loader
    test_loader = run_config.test_loader
    print('*'*8)
    print(len(train_loader))
    print(len(valid_loader))
    print(len(test_loader))
    print('*'*8)

    best_accuracy = 0.0
    num_epochs = 100

    for epoch in range(num_epochs):
        train(epoch, teacher_model, train_loader, optimizer, device)
        # val_accuracy = validate(teacher_model, valid_loader, device)
        test_accuracy = test(teacher_model, test_loader, device)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(teacher_model.state_dict(), 'best_model.pth.tar')

    print(f'Saved Best Model with Test Accuracy: {best_accuracy:.2f}%')

