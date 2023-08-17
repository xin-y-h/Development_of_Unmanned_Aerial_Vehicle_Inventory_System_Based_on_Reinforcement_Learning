import math
from pickletools import optimize
import random
# from typing_extensions import Self
import cv2
from cv2 import resize
# import pupil_apriltags as apriltags 
# import apriltag as apriltags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

import torchvision.models as models



max_action=16
min_action=0



class ResNet (nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()

        # 呼叫模型 pretrained=True
        self.restnet50=models.resnet34(pretrained=True)
        self.restnet50.fc=nn.Linear(512,1024)

        # self.restnet=restnet50()
        self.relu = nn.ReLU()

    def forward(self,x):
        x=self.restnet50(x)
        output=self.relu(x)
        print("ResNet")
        # print("ResNet output",output)
        return output

# Policy架構
class Policy(nn.Module):
    def __init__(self):
        super(Policy,self).__init__()
        # n_feat=16*16*9,n_out=12
        self.fc1 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, max_action+1)#12為n_actions
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

        # 初始化權重參數
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        # fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.out(x)
        action_pro=F.softmax(x,dim=1)
        print("policy")
        return action_pro


# Value架構
class Value(nn.Module):
    def __init__(self):
        super(Value,self).__init__()
        # n_feat=16*16*9,n_out=1
        self.fc1 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)#輸出一個value
        self.relu = nn.ReLU()
        # 初始化權重參數
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        # fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        value = self.out(x)
        print("value")
        return value

# 組裝Actor架構
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.state_train=ResNet()
        self.pol=Policy()
        self.val=Value()
    
    def forward(self,img):
        
        x1=self.state_train(img)
        x2=self.pol(x1)
        x3=self.val(x1)
        print("Actor")
        # print(x3.shape)
        return x2,x3
