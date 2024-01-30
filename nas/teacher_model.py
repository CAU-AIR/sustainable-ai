import os
import random
import argparse
import numpy as np
from PIL import Image
import horovod.torch as hvd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from ofa.utils.run_config import DistributedImageNetRunConfig, DistributedCasiaWebRunConfig

class CASIAWebFaceDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_all_data(dataset_path):
    classes = os.listdir(dataset_path)
    all_files = []

    for cls in classes:
        cls_folder = os.path.join(dataset_path, cls)
        cls_files = [os.path.join(cls_folder, file) for file in os.listdir(cls_folder)]
        all_files += [(file, cls) for file in cls_files]

    return all_files

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
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="face")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    hvd.init()
    num_gpus = hvd.size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")

    expansion = 4

    teacher_model = models.resnet50()
    teacher_model.fc = nn.Linear(512*expansion, 10575)
    teacher_model.to(device)

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    dataset_path = 'dataset/CasiaAligned'
    all_files = get_all_data(dataset_path)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CASIAWebFaceDataset(all_files, transform=transform)
    # test_dataset = CASIAWebFaceDataset(test_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    best_accuracy = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        print('epoch : ', epoch)
        train(epoch, teacher_model, train_loader, optimizer, device)
        # val_accuracy = validate(teacher_model, valid_loader, device)
        # test_accuracy = test(teacher_model, test_loader, device)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(teacher_model.state_dict(), 'best_model.pth.tar')

    print(f'Saved Best Model with Test Accuracy: {best_accuracy:.2f}%')

