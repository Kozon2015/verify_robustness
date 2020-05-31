from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import numpy as np
import os

import argparse
import utils
from torch.autograd import Variable
from single_network.resnet18 import resnet18
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='PyTorch CNN Training')
parser.add_argument('--model', type=str, default='Resnet18', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='CIFAR100', help='CNN architecture')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save_dir', type=str, default='model', help='save log and model')
opt = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_Test_acc = 0  # best PublicTest accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

total_epoch = 80

#path = os.path.join(opt.dataset + '_' + opt.model)
path = os.path.join('single_network', opt.save_dir)
if not os.path.isdir(path):
    os.mkdir(path)

results_log_csv_name = opt.save_dir+'results_log.csv'

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

#train and test data
DATAROOT = '/data/CIFAR100'
train_data = torchvision.datasets.CIFAR100(
    root=DATAROOT, train=True, download=False, transform=transform_train)
test_data = torchvision.datasets.CIFAR100(
    root=DATAROOT, train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=opt.bs, shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)


# Model
net = resnet18()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch,eta_min=0)

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'best_model.pth'))

    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['acc']
    best_Test_acc_epoch = checkpoint['epoch']
    start_epoch = checkpoint['epoch'] + 1
    for x in range(start_epoch):
        scheduler.step()
else:
    print('==> Building model..')

if use_cuda:
    net = net.cuda()


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    scheduler.step()
    print('learning_rate: %s' % str(scheduler.get_lr()))
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))

    Train_acc = 100.*float(correct)/total


def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    net.eval()
    Test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        Test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (Test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    # Save checkpoint.
    Test_acc = 100.*float(correct)/total
    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': Test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'best_model.pth'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch



#record train log
with open(os.path.join(path, results_log_csv_name), 'w') as f:
        f.write(' epoch , train_acc, test_acc, time\n')
#start train
for epoch in range(start_epoch, total_epoch):
    print('current time:',datetime.now().strftime('%b%d-%H:%M:%S'))
    train(epoch)
    test(epoch)
    # Log results
    with open(os.path.join(path, results_log_csv_name), 'a') as f:
        f.write('%03d,%0.3f,%0.3f,%s,\n' % (epoch, Train_acc, Test_acc,datetime.now().strftime('%b%d-%H:%M:%S')))

print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)

# best ACC
with open(os.path.join(path, results_log_csv_name), 'a') as f:
    f.write('%s,%03d,%0.3f,\n' % ('best acc (test)',best_Test_acc_epoch, best_Test_acc))
