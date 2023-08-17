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

from test_AC_env import env
from network import Actor
from test_AC import Actor_Net,n_epochs,lr,lr_decay,gamma,device,batch_size,n_episodes

inputfile='/home/rvl122-4090/Desktop/hsinying/real_dataset/'#test file path
outputfile='./outputpicture/'#output file path
load_path_actor="./weight/AC_model_actor.pth"




train_env=env()
agent=Actor_Net(n_epochs, batch_size, lr, lr_decay, gamma, device)
checkpoint_agent = torch.load(load_path_actor)
agent.Actor.load_state_dict(checkpoint_agent['net'])
# agent.old_pi.load_state_dict(checkpoint_agent['net'])

for j in range(10):
    tag_id_list=np.zeros(50)
    obs=train_env.reset()
    action_to_angle=obs[5]
    reward_totle = 0 
    while not train_env.done:
        action,_, _ = agent.choose_action(obs,action_to_angle)
        print("i_episode_action",action)
        next_obs, reward, done,action_to_angle = train_env.env_step(obs[0],obs[1],obs[2],action,obs[3])
        reward_totle += reward 
        img=cv2.imread(inputfile+str(obs[0])+"_"+str(obs[1])+"_"+str(obs[2])+'_Color.png')
        
        tag_detector = apriltags.Detector()  
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
        cv2.imwrite(outputfile+str(j)+"_"+str(train_env.done_count)+"_"+str(obs[0])+"_"+str(obs[1])+"_"+str(obs[2])+'.jpg',img )

        if done:
            print("done",done)
            print("train_env.done",train_env.done)
            img=cv2.imread(inputfile+str(obs[0])+"_"+str(obs[1])+"_"+str(obs[2])+'_Color.png')
            tag_detector = apriltags.Detector()  
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
            cv2.imwrite(outputfile+str(j)+"_"+str(train_env.done_count)+"_"+str(obs[0])+"_"+str(obs[1])+"_"+str(obs[2])+'.jpg',img )

            img=cv2.imread(inputfile+str(next_obs[0])+"_"+str(next_obs[1])+"_"+str(next_obs[2])+'_Color.png')
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
            cv2.imwrite(outputfile+str(j)+"_"+str(train_env.done_count+1)+"_"+str(next_obs[0])+"_"+str(next_obs[1])+"_"+str(next_obs[2])+'.jpg',img )


            break
        
        obs = next_obs

        

                
    print("Score：", reward_totle) 

    count=sum(tag_id_list)
    print("貨物數量",count)
    # input()
            





    
            


   




    