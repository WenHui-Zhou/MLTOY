'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import config
from models import *
from utils import *

use_cuda = torch.cuda.is_available()
best_acc = 0  # best val accuracy
best_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

train_path = "./data/train/"
val_path = "./data/val/"

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.RandomSizedCrop(224),  #对图片随机切割后，resize成224*224
    transforms.RandomHorizontalFlip(), #对图片进行随机的翻转
    transforms.ToTensor(),   #将图片数据由[0,255]转化为[w,h,c] 格式，范围在[0,1]
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 设置（R,G，B）的平均值，方差
])

# trainset 表示所需的所有数据的数据库，按照transform的格式读取
# torch.utils.data.DataLoader,将dataset封装成一个迭代器，
trainset = torchvision.datasets.ImageFolder(train_path, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

valset = torchvision.datasets.ImageFolder(val_path, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

classes = ('dog', 'cat')

# Model
if config.pretrained:

    model_urls = {
        'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    }
    pretrained_modelname = "densenet121"
    print("Using Pretrained Model: ")
    net = models.__dict__[pretrained_modelname]()  #model中存了VGG,ResNet,AlexNet等
    net.load_state_dict(model_zoo.load_url(model_urls[pretrained_modelname], model_dir="./model_dir/"))
    net.classifier = nn.Linear(1024, 2)  #输入1024 输出2   使用pretrained 的 CNN，添加一层，用于分类
    print("net: ", net)

else:
    if config.resume:  #使用已经训练好保存了的模型
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        best_epoch = checkpoint['epoch']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        #net = VGG('VGG19')
        net = ResNet18()
        #net = GoogLeNet()
        #net = DenseNet121()
        #net = ResNeXt29_2x64d()
        #net = LeNet()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))  #并行地使用gpu
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()  #loss函数
optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4) #momentum 动量，急速收敛，weight_decay = 5e-4 权值衰减，返回值过拟合
#optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()   #将梯度缓存清空
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)  # 数据在这个地方train
        loss = criterion(outputs, targets)
        loss.backward() #loss反向传播，更新权值
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("Epoch: ", epoch, "Acc: ", 100.*correct/total, correct, total)

def val(epoch):

    global best_acc
    global best_epoch

    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  #将 tensor 转移到cuda上运行
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("Epoch: ", epoch, "Acc: ", 100.*correct/total, correct, total)
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            #'net': net.module if use_cuda else net,
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
        best_epoch = epoch
    print("best epoch: ", best_epoch, " acc: ", best_acc)

for epoch in range(start_epoch, start_epoch + config.epochs):
    if epoch < 20:
        train(epoch)
        val(epoch)
    elif epoch >= 20 and epoch < 40:
        optimizer = optim.SGD(net.parameters(), lr=config.lr/10.0, momentum=0.9, weight_decay=5e-4)
        train(epoch)
        val(epoch)
    else:
        optimizer = optim.SGD(net.parameters(), lr=config.lr/100.0, momentum=0.9, weight_decay=5e-4)
        train(epoch)
        val(epoch)


