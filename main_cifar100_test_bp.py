'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
from models import *
from utils import progress_bar
from diffGrad import diffGrad
from Radam import Radam
from AdaBelief import AdaBelief
from AdamInject import AdamInject
from diffGradInject import diffGradInject
from RadamInject import RadamInject
from AdaBeliefInject import AdaBeliefInject
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
#parser.add_argument('--lr', default=0.0001, type=float, help='learning rate'); lr1 = '0001'
parser.add_argument('--lr', default=0.001, type=float, help='learning rate'); lr1 = '001'
#parser.add_argument('--lr', default=0.01, type=float, help='learning rate'); lr1 = '01'
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch

class bitplanetransform (object):
    def __call__(self,inputs):
        inputs = (np.array(inputs))
        # print (inputs.shape)
        # print (inputs)
        # pause
        # print(torch.max(inputs))
        # print(torch.min(inputs))
        # pause
        bitplane_8 = (inputs % 2)
        inputs = (np.floor(inputs / 2))
        bitplane_7 = (inputs % 2)
        inputs = (np.floor(inputs / 2))
        bitplane_6   = (inputs % 2)
        inputs = (np.floor(inputs / 2))
        bitplane_5 = (inputs % 2)
        inputs = (np.floor(inputs / 2))
        bitplane_4 = (inputs % 2)
        inputs = (np.floor(inputs / 2))
        bitplane_3 = (inputs % 2)
        inputs = (np.floor(inputs / 2))
        bitplane_2 = (inputs % 2)
        inputs = (np.floor(inputs / 2))
        bitplane_1 = (inputs % 2)
        
    
        
        #inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s12345678
        # inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_8 * 2 + bitplane_7 * 1  #s12345687
        # inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_6 * 8 + bitplane_5 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s12346578
        # inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_4 * 32 + bitplane_3 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s12435678
        # inputs = bitplane_2 * 128 + bitplane_1 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s21345678
        # inputs = bitplane_1 * 128 + bitplane_3 * 64 + bitplane_2 * 32 + bitplane_4 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s13245678
        # inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_3 * 32 + bitplane_5 * 16 + bitplane_4 * 8 + bitplane_6 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s12354678
        # inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_5 * 8 + bitplane_7 * 4 + bitplane_6 * 2 + bitplane_8 * 1  #s12345768
        # inputs = bitplane_8 * 128 + bitplane_2 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_7 * 2 + bitplane_1 * 1  #s82345671
        # inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_6 * 8 + bitplane_5 * 4 + bitplane_8 * 2 + bitplane_7 * 1  #s12346587
        # inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_4 * 32 + bitplane_3 * 16 + bitplane_6 * 8 + bitplane_5 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s12436578
        # inputs = bitplane_2 * 128 + bitplane_1 * 64 + bitplane_4 * 32 + bitplane_3 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s21435678
        # inputs = bitplane_2 * 128 + bitplane_1 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_8 * 2 + bitplane_7 * 1  #s21345687
        # inputs = bitplane_2 * 128 + bitplane_1 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_6 * 8 + bitplane_5 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s21346578
        # inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_4 * 32 + bitplane_3 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_8 * 2 + bitplane_7 * 1  #s12435687
        inputs = bitplane_1 * 128 + bitplane_2 * 64 + bitplane_4 * 32 + bitplane_3 * 16 + bitplane_6 * 8 + bitplane_5 * 4 + bitplane_8 * 2 + bitplane_7 * 1  #s12436587
        # inputs = bitplane_2 * 128 + bitplane_1 * 64 + bitplane_4 * 32 + bitplane_3 * 16 + bitplane_6 * 8 + bitplane_5 * 4 + bitplane_7 * 2 + bitplane_8 * 1  #s21436578
        # inputs = bitplane_2 * 128 + bitplane_1 * 64 + bitplane_3 * 32 + bitplane_4 * 16 + bitplane_6 * 8 + bitplane_5 * 4 + bitplane_8 * 2 + bitplane_7 * 1  #s21346587
        #inputs = bitplane_2 * 128 + bitplane_1 * 64 + bitplane_4 * 32 + bitplane_3 * 16 + bitplane_5 * 8 + bitplane_6 * 4 + bitplane_8 * 2 + bitplane_7 * 1  #s21435687
        # inputs = bitplane_2 * 128 + bitplane_1 * 64 + bitplane_4 * 32 + bitplane_3 * 16 + bitplane_6 * 8 + bitplane_5 * 4 + bitplane_8 * 2 + bitplane_7 * 1  #s21436587

        

        inputs = Image.fromarray(np.uint8(inputs)).convert('RGB')
        return inputs
        




# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])




transform_test = transforms.Compose([
  bitplanetransform (),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

bs = 64 #set batch size
# trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = VGG('VGG16'); net1 = 'vgg16'
net = ResNet18(); net1 = 'ResNet18'
# net = ResNet50(); net1 = 'ResNet50'
# net = SENet18(); net1 = 'senet18'
# net = ResNet101(); net1 = 'ResNet101'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr); optimizer1 = 'Adam'


path_to_file = './TrainedModel/CIFAR100_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.t7'
# if os.path.exists(path_to_file):
# Load checkpoint.
print('==> loading the trained model..')
assert os.path.isdir('CheckpointsResults'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(path_to_file)
net.load_state_dict(checkpoint['net'])


    
    
    
global acc
net.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
acc = 100. * correct / total
print("Accuracy: ", acc)

