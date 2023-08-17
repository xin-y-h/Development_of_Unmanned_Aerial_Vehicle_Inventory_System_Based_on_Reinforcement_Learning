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


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import yolov5
import pandas as pd



inputfile='/home/rvl122-4090/Desktop/hsinying/real_dataset/'#test file path

random.seed()#每次random都不一樣(給參數就會變成每次random給的數都一樣)
tag_number=16 #Apriltag數量
done_count_num=50
threshold=3

class env():
    def __init__(self):
        # 相機參數
        self.cameraParams_Intrinsic = [960, 960, 960, 540]  # camera_fx, camera_fy, camera_cx, camera_cy 
        self.camera_matrix = np.array(([960, 0, 960],
                                [0, 960, 540],
                                [0, 0, 1.0]), dtype=np.double)

        self.done = False
        self.done_count=0
        self.step_count=0
        self.tag_id_list=np.zeros(500)
        self.action_to_angle=False


    # 重新設定環境（產生初始狀態值)
    def reset(self):
        
        self.action_to_angle=False
        angle=random.randrange(-40,41,10)
        
        self.done=False
        self.done_count=0
        self.step_count=0
        self.tag_id_list=np.zeros(500)


        firstscenes=random.randrange(0,2,1)
        firstscenes=0
        if firstscenes==0:
            x=2
            y=-1
            
        elif firstscenes==1:
            x=12
            y=1


        img=self.img_resize(x,y,angle)
        _,_,self.done=self.area_value_singl(x,y,angle,self.done)
        print("reset")
        return x,y,angle,self.done,img,self.action_to_angle
    
    # resize照片
    def img_resize(self,x,y,angle):
        
        img=cv2.imread(inputfile+str(x)+"_"+str(y)+"_"+str(angle)+'_Color.png')
        print(np.shape(img))
        print(x,y,angle)
        resize_img=cv2.resize(img,(960,540))
        resize_img=resize_img.transpose(2,1,0)
        # print(resize_img)
        # print(np.shape(resize_img))
        resize_img=torch.tensor(resize_img, dtype=torch.float32).unsqueeze(0)
        resize_img /= 255.0
        # resize_img=resize_img.float()
        resize_img=resize_img.to("cuda")
        # print(resize_img)
        return resize_img

    

    # 現在狀態與下一狀態畫面佔比的reward計算
    def reward_area_value(self,x,y,angle,done,x_next,y_next,angle_next,done_next):
        now,_,_=self.area_value_singl(x,y,angle,done)
        next,_,done=self.area_value_singl(x_next,y_next,angle_next,done_next)
        if now or next!=0:
           
            if(now==next):#no move
                reward_area=-now
            #(check is next is better than now)
            reward_area=next-now
        else:
            reward_area=-0.0001
        return reward_area*10,done

        

# 單張畫面佔比的計算
    def area_value_singl(self,x,y,angle,done):
        
        img=cv2.imread(inputfile+str(x)+"_"+str(y)+"_"+str(angle)+'_Color.png')
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
       
        tag_detector = apriltags.Detector()  # Build a detector for apriltag
        
        tags = tag_detector.detect(gray)
        tag_count=0
        # 計算tag在畫面中的面積佔比(取畫面中有偵測到的最高值)
        if(tags != []):
            # print("5")
            area=[]
            for tag in tags:
                if(self.tag_id_list[tag.tag_id]!=1):
                    a=tag.corners[3][0]*tag.corners[2][1]+tag.corners[2][0]*tag.corners[1][1]+tag.corners[1][0]*tag.corners[0][1]+tag.corners[0][0]*tag.corners[3][0]
                    b=tag.corners[3][1]*tag.corners[2][0]+tag.corners[2][1]*tag.corners[1][0]+tag.corners[1][1]*tag.corners[0][0]+tag.corners[0][1]*tag.corners[3][0]
                    aera=abs(a-b)/2
                    # x0=tags.corners[3][0]
                    # y0=tags.corners[3][1]
                    # x1=tags.corners[2][0]
                    # y1=tags.corners[2][1]
                    # x2=tags.corners[1][0]
                    # y2=tags.corners[1][1]
                    # x3=tags.corners[0][0]
                    # y3=tags.corners[0][1]
                    # 任意四點面積：a=x0y1+x1y2+x2y3+x3y0
                    # b=y0x1+y1x2+y2x3+y3x0
                    # Area=|a-b|/2
                    # area.append(aera/(self.cameraParams_Intrinsic[2]*self.cameraParams_Intrinsic[3]))
                    area.append(aera/(self.cameraParams_Intrinsic[2]*self.cameraParams_Intrinsic[3]))
                    # print("6")
                    # print(area)
                # area_value=max(area)
                    self.tag_id_list[tag.tag_id]=1
                    tag_count=tag_count+1
                    print("id=",tag.tag_id)
                    print("area_value_singl_x,y,angle",x,y,angle)
                if area!=[]:
                    # area_value=sum(area)
                    area_value=sum(area)/tag_count
                    # area_value=max(area)
                else:
                    area_value=0
            
            if sum(self.tag_id_list)==tag_number:
                done=True
            # done=True
            print("area_value_singl—done-----------------------------------------------========///==",area_value)
            print("self.tag_id_list",self.tag_id_list)
            print("sum(self.tag_id_list)",sum(self.tag_id_list))
            print("done",done)
            print("self.done",self.done)
            # input()
        if(tags == []):
            # print("7")
            area_value=0
        
        print("area_value_singl")
        return area_value,tag_count,done


    def k_mean_center(self,x,y,angle):
        img=cv2.imread(inputfile+str(x)+"_"+str(y)+"_"+str(angle)+'_Color.png')
        model = yolov5.load('./exp8/weights/last.pt') #yolo weight path
        model.conf = 0.5  # NMS confidence threshold
        
        results = model(img)
        
        yolo_predictions = results.pred[0]
        
        x1=yolo_predictions[:, :1].tolist()
        y1=yolo_predictions[:, 1:2].tolist()
        x2=yolo_predictions[:, 2:3].tolist()
        y2=yolo_predictions[:, 3:4].tolist()
        print("x1_all",x1)
        print("x2_all",x2)
        print("y1_all",y1)
        print("y2_all",y2)
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        tag_detector = apriltags.Detector()  # Build a detector for apriltag
        # print("3")
        tags = tag_detector.detect(gray)
        print("tag****",len(tags))

        if(tags != []):
            for tag in tags:
                for i in range(len(x1)-1,-1,-1):
                    if((x1[i]<=(tag.corners[0][0]+tag.corners[2][0])/2<=x2[i]) & (y1[i]<=(tag.corners[0][1]+tag.corners[2][1])/2<=y2[i])):
                        del(x1[i])
                        del(x2[i])
                        del(y1[i])
                        del(y2[i])
                        break

        center=[]
        for i in range(len(x1)):
            center.append([(np.squeeze(x1[i])+np.squeeze(x2[i]))/2,np.squeeze((y1[i])+np.squeeze(y2[i]))/2])
        if(center==[]):
            center=[[480,270]]

        
        dx=np.array(center)
        center=[]
        print("dx",dx)

        # 去除離群點
        dx_pd=pd.DataFrame(dx,columns=["x","y"])
        
        k_list=[]
        # dy=np.array(box1.tolist())
        rel_dis=[]
        # 用 KMeans 在資料中找出 n 個分組
        kmeans = KMeans(n_clusters=1)#max_iter=迭代次數
        kmeans.fit(dx)#開始聚類
        distance=pd.DataFrame(dx-kmeans.cluster_centers_,columns=["x","y"])
        print("kmeans.cluster_centers_",kmeans.cluster_centers_)
        print("distance",distance)
        abs_dis=distance.apply(np.linalg.norm,axis=1)#求絕對距離，apply對資料做匹量處理：np.linalg.norm範數(平方相加開根號)
        rel_dis.append(abs_dis/abs_dis.median())
        dx_pd["rel_dis"]=pd.concat(rel_dis)
        out=dx_pd.rel_dis.apply(lambda x:1 if x>threshold else 0)
        print("out",out)
        print(len(out.index))
        for i in range(0,len(out.index)):
            if(out[i]==0):
                print("-------",out.index[i]) 
                k_list.append(out.index[i])
        print("del_list",k_list) 
        for i in range (len(k_list)) :
            center.append(dx[k_list[i]]) 
        print("new_center",center) 

        dx_new=np.array(center) 
        kmeans.fit(dx_new)#開始聚類   

        # # 沒去除離群點
        # kmeans = KMeans(n_clusters=1)#max_iter=迭代次數
        # kmeans.fit(dx)#開始聚類
                
        print("kmeans.cluster_centers_",kmeans.cluster_centers_)
        return kmeans.cluster_centers_


    #選擇動作後送入env的輸出以及到達目標判讀
    def env_step(self,x,y,angle,action,done):

        # print("obs",x,y,angle,action,done)
        # input()

        self.done_count=self.done_count+1
        if self.done_count>done_count_num:
            self.done=True

        next_observation=self.next_state(x,y,angle,action,done)
        print(" ")
        print("action",action)
        print(" ")
        
        self.step_count=self.step_count+1
        
        print(next_observation[0],next_observation[1],next_observation[2])
        print(x,y,angle)

        
        k_mean_center_now=self.k_mean_center(x,y,angle)
        

        
        # 下一張單張面積佔比+tag數量
        _,reward_tag_count,done_n=self.area_value_singl(next_observation[0],next_observation[1],next_observation[2],next_observation[3])
        
        
        print("self.done",self.done)
       
        reward=0.5*(reward_tag_count/tag_number)


        # 中心點_移動動作選擇＋角度
        # 左
        if(np.squeeze(k_mean_center_now[:,0])>(self.cameraParams_Intrinsic[2]/2)):
            if(action==0 or action==4 or action==6 or 8<=action<=11):
                reward=reward+0.3*(((np.squeeze(k_mean_center_now[:,0]))-(self.cameraParams_Intrinsic[2]/2))/(self.cameraParams_Intrinsic[2]/2))
        # 右
        elif(np.squeeze(k_mean_center_now[:,0])<(self.cameraParams_Intrinsic[2]/2)):
            if(action==2 or action==5 or action==7 or 13<=action<=16):
                reward=reward+0.3*((-(np.squeeze(k_mean_center_now[:,0]))+(self.cameraParams_Intrinsic[2]/2))/(self.cameraParams_Intrinsic[2]/2)) 
   
       # 找不到新tag依照步數扣分
        if reward_tag_count>0:
            self.step_count=0
            self.action_to_angle=False
        if reward_tag_count==0:
            self.action_to_angle=True
            reward=reward-(self.step_count/done_count_num)*0.5
            if self.step_count>=4:
                self.action_to_angle=False # 轉太多次會卡住，因此多次後再改回移動
                if self.step_count%3==0:
                    self.action_to_angle=True
        # 所有tag皆掃完
        if done_n ==True:
            self.done=done_n 
            reward=reward+1

        print("env_step")
        
        return next_observation,reward,self.done,self.action_to_angle


    #此動作產生下一個狀態
    def next_state(self,x,y,angle,action_num,done):
              
        # 移動
        if (action_num==0):
            x_next=x+1
            angle_next=angle
            y_next=y
        
        elif (action_num==1):
            y_next=y+1
            angle_next=angle
            x_next=x            

        elif (action_num==2) :
            x_next=x-1
            angle_next=angle
            y_next=y

        elif (action_num==3):
            y_next=y-1
            angle_next=angle
            x_next=x

        elif (action_num==4) :
            x_next=x+1
            y_next=y+1
            angle_next=angle
            
        elif (action_num==5) :
            x_next=x-1
            y_next=y+1
            angle_next=angle

        elif (action_num==6) :
            x_next=x+1
            y_next=y-1
            angle_next=angle
            
        elif (action_num==7) :
            x_next=x-1
            y_next=y-1
            angle_next=angle

        # 角度
        # elif (action_num==8)  & (x!=0):

        elif (action_num==8):
            angle_next=-40
            x_next=x
            y_next=y

        elif(action_num==9) :
            angle_next=-30
            x_next=x
            y_next=y

        elif(action_num==10) :
            angle_next=-20
            x_next=x
            y_next=y
        
        elif(action_num==11) :
            angle_next=-10
            x_next=x
            y_next=y

        elif(action_num==12) :
            angle_next=0
            x_next=x
            y_next=y
        
        elif(action_num==13) :
            angle_next=10
            x_next=x
            y_next=y
        
        elif(action_num==14) :
            angle_next=20
            x_next=x
            y_next=y

        elif(action_num==15) :
            angle_next=30
            x_next=x
            y_next=y

        elif(action_num==16) :
            angle_next=40
            x_next=x
            y_next=y

        img=self.img_resize(x_next,y_next,angle_next)

        print("next_state")
        
        return x_next,y_next,angle_next,done,img
