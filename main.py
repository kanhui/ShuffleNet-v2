# -*- coding: utf-8 -*-

"""
@Date: 2018/12/25

@Author: dreamhomes

@Summary: model train and test
"""
import os
import time

from utils import progress_bar
from skimage import io

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from shufflenet_v2 import ShuffleNetV2


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Directory with all the images.
        :param transform:
        """
        images = []
        labels = []
        for img_name in os.listdir(root_dir):
            label = int(img_name.split('_')[0])
            labels.append(label)

            image = io.imread(os.path.join(root_dir, img_name))
            images.append(image)
        # print(len(images), len(labels))

        self.images = images
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dir = 'D:/MachineLearning/datasets/cifar-10/train_cifar10'
test_dir = 'D:/MachineLearning/datasets/cifar-10/test_cifar10'

train_data = MyDataset(root_dir=train_dir, transform=transform)
test_data = MyDataset(root_dir=test_dir, transform=transform)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

best_accuracy = 0.0

net = ShuffleNetV2(num_classes=10)
if torch.cuda.is_available():
    net.cuda()

resume = False
if resume:
    print('\n==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ShuffleNet_V2.pth')
    net.load_state_dict(checkpoint['net'])
    best_accuracy = checkpoint['acc']

optimizer = torch.optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train(epoch):
    """
    model train
    :param epoch:
    :return:
    """
    print('\nEpoch: %d' % epoch)
    correct = 0
    train_loss = 0.0
    total = 0
    net.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        scores = net.forward(data)
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = scores.max(1)[1]
        total += label.size(0)
        correct += pred.eq(label).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Train_loss: %.3f | Train_acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_acc = 100. * correct / float(len(train_loader.dataset))
    return train_acc


def validate():
    """
    mode validate
    :param epoch:
    :return:
    """
    global best_accuracy
    correct = 0
    val_loss = 0.0
    total = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            scores = net.forward(data)
            loss = criterion(scores, label)
            val_loss += loss.item()
            pred = scores.max(1)[1]
            total += label.size(0)
            correct += pred.eq(label).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Val_loss: %.3f | Val_acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    val_acc = correct / float(total) * 100.

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        state = {
            'net': net.state_dict(),
            'acc': val_acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, "./checkpoint/ShuffleNet_V2.pth")
    return val_acc


def test():
    """
    model test
    :return:
    """
    net.load_state_dict(torch.load("./checkpoint/ShuffleNet_V2.pth")['net'])
    net.eval()
    test_correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            scores = net(data)
            pred = scores.data.max(1)[1]
            test_correct += pred.eq(label).sum().item()

    return 100. * test_correct / float(len(test_loader.dataset))

epochs = 50
train_accuracy, val_accuracy = [], []
for i in range(1, epochs+1):
    train_accuracy.append(train(i))
    val_accuracy.append(validate())

test_acc = test()
print('\nTest accuracy on CIFAR-10 is {:.2f}%\n'.format(test_acc))
