import math
from pickletools import optimize
import random
# from typing_extensions import Self
import cv2
from cv2 import resize
# import pupil_apriltags as apriltags 
import apriltag as apriltags
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
        


# Policy架構(Actor_Net)
class Actor_Net(nn.Module):
    def __init__(self):
        super(Actor_Net,self).__init__()
        # n_feat=16*16*9,n_out=10
        self.state_train=ResNet()
        self.fc1 = nn.Linear(1024, 512)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(512, 1)#10為n_actions
        self.out.weight.data.normal_(0,0.1)
        self.std_out = nn.Linear(512, 1)
        self.std_out.weight.data.normal_(0, 0.1)
    
    def forward(self, state_img):
        # fully connected layers
        inputstate=self.state_train(state_img)
        inputstate = self.fc1(inputstate)
        inputstate = F.tanh(inputstate)
        mean=max_action*torch.tanh(self.out(inputstate))#输出概率分布的均值mean
        std=F.softplus(self.std_out(inputstate))#softplus激活函数的值域>0
        return mean,std
        


# Value架構(Critic_Net)
class Critic_Net(nn.Module):
    def __init__(self):
        super(Critic_Net,self).__init__()
        # n_feat=16*16*9,n_out=1
        self.state_train=ResNet()
        self.fc1 = nn.Linear(1024, 512)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(512, 1)#輸出一個value
        self.out.weight.data.normal_(0,0.1)
        self.relu = nn.ReLU()
        
        
    def forward(self, state_img):
        # fully connected layers
        inputstate=self.state_train(state_img)
        inputstate = self.fc1(inputstate)
        inputstate = self.relu(inputstate)
        Q_value = self.out(inputstate)
        print("Q_value")
        return Q_value