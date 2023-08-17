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
import pyshine as ps

from test_env import env
from network import Actor_Net,Critic_Net
from test_ppo import Actor,Critic,EP_LEN,device,batch_size

inputfile='/home/rvl122-4090/Desktop/hsinying/AC/getpicture/shelf_dataset_1/'
outputfile='./simulation_step150/'
load_path_actor="./PPO2_model_actor_last.pth"
load_path_critic="./PPO2_model_critic_last.pth"

num=[]


train_env=env()
actor=Actor(device,batch_size)
critic=Critic(device)
checkpoint_actor = torch.load(load_path_actor)
actor.old_pi.load_state_dict(checkpoint_actor['net'])
checkpoint_critic = torch.load(load_path_critic)
critic.critic_v.load_state_dict(checkpoint_critic['net'])
for j in range(10):
    tag_id_list=np.zeros(100)
    state = train_env.reset()
    scenes=state[5]
    action_to_angle=state[6]
    total_rewards = 0
    detect_tag=[]
    
    

    for timestep in range(EP_LEN): 
        action, action_logprob = actor.choose_action(state,scenes,action_to_angle)
        next_state, reward, done,action_to_angle = train_env.env_step(state[0],state[1],state[2],action,state[3]) # 执行动作
        total_rewards += reward

        img=cv2.imread(inputfile+str(state[0])+"_"+str(state[1])+"_"+str(state[2])+'.jpg')
        
        tag_detector = apriltags.Detector()  # Build a detector for apriltag
        gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        tags = tag_detector.detect(gray)
        for tag in tags:
            if(tag_id_list[int(tag.tag_id)]!=1):
                cv2.circle(img, tuple(tag.corners[0].astype(int)), 4,(255,0,0), 2) # left-top
                cv2.circle(img, tuple(tag.corners[1].astype(int)), 4,(255,0,0), 2) # right-top
                cv2.circle(img, tuple(tag.corners[2].astype(int)), 4,(255,0,0), 2) # right-bottom
                cv2.circle(img, tuple(tag.corners[3].astype(int)), 4,(255,0,0), 2) # left-bottom
                tag_id_list[int(tag.tag_id)]=1
        count=sum(tag_id_list)
        text = 'Apriltag:'+str(count)

        # 加入文字方塊
        img = ps.putBText(
                img,                            # 原始影像
                text,                             # 文字內容
                text_offset_x = 20,               # X 軸座標
                text_offset_y = 20,               # Y 軸座標
                vspace = 20,                      # 縱向空間
                hspace = 20,                      # 橫向空間
                font_scale = 1.0,                 # 字型大小
                background_RGB = (228, 225, 222), # 背景顏色
                text_RGB = (255, 90, 150)           # 文字顏色 
            )
        cv2.imwrite(outputfile+str(j)+"_"+str(timestep)+"_"+str(state[0])+"_"+str(state[1])+"_"+str(state[2])+'.jpg',img )
        if done:
            img=cv2.imread(inputfile+str(next_state[0])+"_"+str(next_state[1])+"_"+str(next_state[2])+'.jpg')
            tag_detector = apriltags.Detector()  # Build a detector for apriltag
            gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            tags = tag_detector.detect(gray)
            for tag in tags:
                if(tag_id_list[int(tag.tag_id)]!=1):
                    cv2.circle(img, tuple(tag.corners[0].astype(int)), 4,(255,0,0), 2) # left-top
                    cv2.circle(img, tuple(tag.corners[1].astype(int)), 4,(255,0,0), 2) # right-top
                    cv2.circle(img, tuple(tag.corners[2].astype(int)), 4,(255,0,0), 2) # right-bottom
                    cv2.circle(img, tuple(tag.corners[3].astype(int)), 4,(255,0,0), 2) # left-bottom
                    tag_id_list[int(tag.tag_id)]=1

            count=sum(tag_id_list)
            text = 'Apriltag:'+str(count)
            # text ='hello world'

            # 加入文字方塊
            img = ps.putBText(
                    img,                            # 原始影像
                    text,                             # 文字內容
                    text_offset_x = 20,               # X 軸座標
                    text_offset_y = 20,               # Y 軸座標
                    vspace = 20,                      # 縱向空間
                    hspace = 20,                      # 橫向空間
                    font_scale = 1.0,                 # 字型大小
                    background_RGB = (228, 225, 222), # 背景顏色
                    text_RGB = (255, 90, 150)           # 文字顏色 
                )
            cv2.imwrite(outputfile+str(j)+"_"+str(timestep+1)+"_"+str(next_state[0])+"_"+str(next_state[1])+"_"+str(next_state[2])+'.jpg',img )
            break
        
        state = next_state


    print("Score：", total_rewards) 

    count=sum(tag_id_list)
    num.append(count)
    print("貨物數量",count)
    # input()




    