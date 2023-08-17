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

from test_env import env
from network import Actor_Net,Critic_Net

from tensorboardX import SummaryWriter
from datetime import datetime
# tensorboard --logdir=runs

outputfile='/home/rvl122/getpicture/outputpicture/'
time = datetime.now()
path = f'runs/{time.strftime("%Y_%m_%d_%H_%M_%S"+"angle_fix_to_change")}'
writer = SummaryWriter(path)


random.seed()#每次random都不一樣(給參數就會變成每次random給的數都一樣)

max_action=16
min_action=0

device="cuda"
EP_MAX = 2000
EP_LEN = 150  #from env import env,inputfile,EP_LEN
GAMMA = 0.99
A_LR = 0.02
C_LR = 0.03
batch_size=8
A_UPDATE_STEPS = 20
C_UPDATE_STEPS = 20
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 
][1]        


# 暫存資料
class MemoryBuffer():
    def __init__(self, batch_size):
        # self.states = []
        self.states_other = []
        self.states_img = []
        
        self.actions = []
        self.rewards = []
        self.dones = []
        self.action_logprob=[]

       
        self.batch_size = batch_size

    def store(self,states_other,states_img, action, reward, done, action_logprob):
        self.states_other.append(states_other)
        self.states_img.append(states_img)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_logprob.append(action_logprob)
        print("store")

    def clear(self,):
        self.states_other = []
        self.states_img = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.action_logprob=[]

        print("clear")
        
    def generate_batch(self,):


        return np.array(list(self.states_other), dtype=np.float32),np.array(list(self.states_img), dtype=np.float32),np.array(list(self.actions), dtype=np.float32), np.array(list(self.rewards),dtype=np.float32),\
               np.array(list(self.dones), dtype=np.float32),np.array(list(self.action_logprob),dtype=np.float32)

    def __len__(self):
        return len(self.states_img)


class Actor():
    def __init__(self,device,batch_size):
        self.batch_size = batch_size
        self.device= torch.device(device)
        self.old_pi,self.new_pi=Actor_Net().to(self.device),Actor_Net().to(self.device)#均值mean

        self.optimizer=torch.optim.Adam(self.new_pi.parameters(),lr=A_LR,weight_decay=0.0001)
        self.memory = MemoryBuffer(self.batch_size)

    def remember(self, states_other,states_img, action, reward, done,logprob):
        
        self.memory.store(states_other,states_img, action, reward, done, logprob)
    
    def batch_remember(self):
        states_other_np,states_img_np,action_np, rewards_np, dones_np,action_logprob_np=self.memory.generate_batch()
        return states_other_np,states_img_np,action_np, rewards_np, dones_np,action_logprob_np
    
    def actor_clear(self):
        self.memory.clear()

    def lr_decay(self):
        self.optimizer.param_groups[0]["lr"] *=0.9 #呼叫時每次都*一個 <1 的數來下降

        
    
    def choose_action(self,obs,scenes_num,action_to_angle):

        inputstate1 = torch.tensor(obs[4]).to(self.device)

        mean,std=self.old_pi(inputstate1)
        mean=torch.nan_to_num(mean,nan=math.exp(-100))
        std=torch.nan_to_num(std,nan=math.exp(-100))
        dist = torch.distributions.Normal(mean, std)
        action=dist.sample()#以分佈曲線取樣
        action=torch.clamp(action,min_action,max_action)#剪裁
        angle_change=random.randrange(0,8,1)
        move_change=random.randrange(0,3,1)
        move_change_2=random.randrange(0,5,1)
        if(action-int(action)>=0.5):
            action=torch.tensor(math.ceil(float(action)))
        else:
            action=torch.tensor(math.floor(float(action)))
     
        if (scenes_num==0):
            pass

        elif(scenes_num==1):
            if (not action_to_angle):
  
                if(obs[0]==35 and (action==0 or action==4 or action==6)):

                    if(obs[1]==0):
                        if move_change==0:
                            action=2
                        if move_change==1:
                            action=3
                        if move_change==2:
                            action=7

                    elif(obs[1]==-2):
                        if move_change==0:
                            action=1
                        if move_change==1:
                            action=2
                        if move_change==2:
                            action=5
                    
                    else:
                        if move_change_2==0:
                            action=1
                        if move_change_2==1:
                            action=2
                        if move_change_2==2:
                            action=3
                        if move_change_2==3:
                            action=5
                        if move_change_2==4:
                            action=7
                    
                elif(obs[0]==2  and (action==2 or action==5 or action==7)):
                    
                    if(obs[1]==0):
                        if move_change==0:
                            action=0
                        if move_change==1:
                            action=3
                        if move_change==2:
                            action=6

                    elif(obs[1]==-2):
                        if move_change==0:
                            action=0
                        if move_change==1:
                            action=1
                        if move_change==2:
                            action=4
                    else:
                        if move_change_2==0:
                            action=0
                        if move_change_2==1:
                            action=1
                        if move_change_2==2:
                            action=3
                        if move_change_2==3:
                            action=4
                        if move_change_2==4:
                            action=6

                elif(obs[1]==0 and (action==1 or action==4 or action==5)):
                    if( obs[0]==35):
                        if move_change==0:
                            action=2
                        if move_change==1:
                            action=3
                        if move_change==2:
                            action=7

                    elif(obs[0]==2):
                        if move_change==0:
                            action=0
                        if move_change==1:
                            action=3
                        if move_change==2:
                            action=6
                    else:
                        if move_change_2==0:
                            action=0
                        if move_change_2==1:
                            action=2
                        if move_change_2==2:
                            action=3
                        if move_change_2==3:
                            action=6
                        if move_change_2==4:
                            action=7
                    
                elif(obs[1]==-2 and (action==3 or action==6 or action==7)):
                    if( obs[0]==35):
                        if move_change==0:
                            action=1
                        if move_change==1:
                            action=2
                        if move_change==2:
                            action=5
                    elif(obs[0]==2):
                        if move_change==0:
                            action=0
                        if move_change==1:
                            action=1
                        if move_change==2:
                            action=4
                    else:
                        if move_change_2==0:
                            action=0
                        if move_change_2==1:
                            action=1
                        if move_change_2==2:
                            action=2
                        if move_change_2==3:
                            action=4
                        if move_change_2==4:
                            action=5

            
            else:
                action=random.randrange(8,17,1)
                
                if(obs[2]==-40 and action==8):
                    if angle_change==0:
                        action=9
                    if angle_change==1:
                        action=10
                    if angle_change==2:
                        action=11
                    if angle_change==3:
                        action=12
                    if angle_change==4:
                        action=13
                    if angle_change==5:
                        action=14
                    if angle_change==6:
                        action=15
                    if angle_change==7:
                        action=16
                    
                    # input()
                
                elif(obs[2]==-30 and action==9):
                    if angle_change==0:
                        action=8
                    if angle_change==1:
                        action=10
                    if angle_change==2:
                        action=11
                    if angle_change==3:
                        action=12
                    if angle_change==4:
                        action=13
                    if angle_change==5:
                        action=14
                    if angle_change==6:
                        action=15
                    if angle_change==7:
                        action=16
                    # input()

                elif(obs[2]==-20 and action==10):
                    if angle_change==0:
                        action=8
                    if angle_change==1:
                        action=9
                    if angle_change==2:
                        action=11
                    if angle_change==3:
                        action=12
                    if angle_change==4:
                        action=13
                    if angle_change==5:
                        action=14
                    if angle_change==6:
                        action=15
                    if angle_change==7:
                        action=16
                    # input()
                
                elif(obs[2]==-10 and action==11):
                    if angle_change==0:
                        action=8
                    if angle_change==1:
                        action=9
                    if angle_change==2:
                        action=10
                    if angle_change==3:
                        action=12
                    if angle_change==4:
                        action=13
                    if angle_change==5:
                        action=14
                    if angle_change==6:
                        action=15
                    if angle_change==7:
                        action=16
                    # input()
                
                elif(obs[2]==0 and action==12):
                    if angle_change==0:
                        action=8
                    if angle_change==1:
                        action=9
                    if angle_change==2:
                        action=10
                    if angle_change==3:
                        action=11
                    if angle_change==4:
                        action=13
                    if angle_change==5:
                        action=14
                    if angle_change==6:
                        action=15
                    if angle_change==7:
                        action=16
                
                elif(obs[2]==10 and action==13):
                    if angle_change==0:
                        action=8
                    if angle_change==1:
                        action=9
                    if angle_change==2:
                        action=10
                    if angle_change==3:
                        action=11
                    if angle_change==4:
                        action=12
                    if angle_change==5:
                        action=14
                    if angle_change==6:
                        action=15
                    if angle_change==7:
                        action=16
                
                elif(obs[2]==20 and action==14):
                    if angle_change==0:
                        action=8
                    if angle_change==1:
                        action=9
                    if angle_change==2:
                        action=10
                    if angle_change==3:
                        action=11
                    if angle_change==4:
                        action=12
                    if angle_change==5:
                        action=13
                    if angle_change==6:
                        action=15
                    if angle_change==7:
                        action=16

                elif(obs[2]==30 and action==15):
                    if angle_change==0:
                        action=8
                    if angle_change==1:
                        action=9
                    if angle_change==2:
                        action=10
                    if angle_change==3:
                        action=11
                    if angle_change==4:
                        action=12
                    if angle_change==5:
                        action=13
                    if angle_change==6:
                        action=14
                    if angle_change==7:
                        action=16

                elif(obs[2]==40 and action==16):
                    if angle_change==0:
                        action=8
                    if angle_change==1:
                        action=9
                    if angle_change==2:
                        action=10
                    if angle_change==3:
                        action=11
                    if angle_change==4:
                        action=12
                    if angle_change==5:
                        action=13
                    if angle_change==6:
                        action=14
                    if angle_change==7:
                        action=15
                    # input()

        action=torch.tensor(action)
        action_logprob=dist.log_prob(action)
        return action.detach(),action_logprob.detach()

    def update_oldpi(self):
        self.old_pi.load_state_dict(self.new_pi.state_dict())
    
    def learn(self,bs1,ba,adv,bap):
        bs1=torch.tensor(bs1).to(self.device)
        ba = torch.tensor(ba).to(self.device)
        adv = torch.tensor(adv).to(self.device)
        bap = torch.tensor(bap).to(self.device)
        for _ in range(A_UPDATE_STEPS):
            mean, std = self.new_pi(bs1)
            
            mean=torch.nan_to_num(mean,nan=math.exp(-100))
            std=torch.nan_to_num(std,nan=math.exp(-100))

            dist_new=torch.distributions.Normal(mean, std)
            
            action_new_logprob=dist_new.log_prob(ba)
            ratio=torch.exp(action_new_logprob - bap.detach())
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - METHOD['epsilon'], 1 + METHOD['epsilon']) * adv
            loss = -torch.min(surr1, surr2)
            loss=loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.new_pi.parameters(), 0.5)
            self.optimizer.step()

class Critic():
    def __init__(self,device):
        self.device= torch.device(device)
        self.critic_v=Critic_Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=C_LR,weight_decay=0.0001) #weight decay防止過擬合，eps是lr衰減限制
        self.lossfunc = nn.MSELoss()

    def lr_decay(self):
        self.optimizer.param_groups[0]["lr"] *=0.95 #呼叫時每次都*一個 <1 的數來下降

    # 預測的評估值
    def get_v(self,bs1):
        bs1=torch.tensor(bs1).to(self.device)
        return self.critic_v(bs1)
    
    # critic學習
    def learn(self,bs1,br):
        bs1=torch.tensor(bs1).to(self.device)
        reality_v = torch.tensor(br).to(self.device)
        for _ in range(C_UPDATE_STEPS):
            v=self.get_v(bs1)
            td_e = self.lossfunc(reality_v, v)
            self.optimizer.zero_grad()
            td_e.backward()
            nn.utils.clip_grad_norm_(self.critic_v.parameters(), 0.5)
            self.optimizer.step()
        return (reality_v-v).detach()


def main():
    train_env=env()
    print("Create an Environment ...")
    actor=Actor(device,batch_size)
    print("Create an actor ...") 
    critic=Critic(device)
    print("Create a critic ...")
   
   
    all_ep_r = []

    for i_episode in range(EP_MAX):   
        obs=train_env.reset()
        scenes_num=obs[5]
        action_to_angle=obs[6]  
        reward_totle=0 
        if i_episode % 100==0:
            actor.lr_decay()
            critic.lr_decay()
            
        
        for timestep in range(EP_LEN):
            print("train_env",i_episode)
            
            action,action_logprob=actor.choose_action(obs,scenes_num,action_to_angle)
            
            next_obs, reward, done ,action_to_angle= train_env.env_step(obs[0],obs[1],obs[2],action,obs[3])

            
            actor.remember(torch.tensor(obs[0:4]).unsqueeze(0).tolist(),obs[4].tolist(),action.tolist(),\
                reward, done,action_logprob.squeeze(0).tolist())

            #PPO 更新
            if ((timestep+1) % batch_size == 0) or (timestep == EP_LEN-1 )or(not done):
                states_other_np,states_img_np,action_np, rewards_np, _,action_logprob_np=actor.batch_remember()
                
                v_observation_ = critic.get_v(torch.tensor(next_obs[4]))
                discounted_r = []

                for reward_i in list(rewards_np)[::-1]:
                    v_observation_ = reward_i + GAMMA * v_observation_
                    discounted_r.append(v_observation_)
                discounted_r.reverse()

                bs1, _, ba, br,bap = np.vstack(states_img_np),np.vstack(states_other_np),action_np,discounted_r,action_logprob_np
                actor.actor_clear()

                advantage=critic.learn(bs1,br)#critic部分更新
                actor.learn(bs1,ba,advantage,bap)#actor部分更新
                actor.update_oldpi()  # pi-new的参数赋给pi-old
                print("ppo_update**************************************************************************")

          
            if done:
                print("done_break******************************************************",done)
                break
            
            obs=next_obs
            reward_totle+=reward

            
        if i_episode == 0:
            all_ep_r.append(reward_totle)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + reward_totle * 0.1)
        print("\rEp: {} |rewards: {}".format(i_episode, reward_totle), end="")
        print("all_ep_r",all_ep_r[i_episode])
        # input()
        # 保存參數
        if i_episode % 5 == 0 and i_episode >= 50:
            save_data1 = {'net': actor.old_pi.state_dict(), 'opt': actor.optimizer.state_dict(), 'i': i_episode}
            save_data2 = {'net': critic.critic_v.state_dict(), 'opt': critic.optimizer.state_dict(), 'i': i_episode}
            if(i_episode==50 or all_ep_r[i_episode]>all_ep_r[i_episode-50]):
                torch.save(save_data1, "./weight/PPO2_model_actor.pth")
                torch.save(save_data2, "./weight/PPO2_model_critic.pth")
            
        if(i_episode==EP_MAX-1):
            save_data1 = {'net': actor.old_pi.state_dict(), 'opt': actor.optimizer.state_dict(), 'i': i_episode}
            save_data2 = {'net': critic.critic_v.state_dict(), 'opt': critic.optimizer.state_dict(), 'i': i_episode}
            torch.save(save_data1, "./weight/PPO2_model_actor_last.pth")
            torch.save(save_data2, "./weight/PPO2_model_critic_last.pth")

        writer.add_scalar('episode reward', reward_totle,i_episode)

        writer.add_scalar('averaged episode reward', all_ep_r[i_episode],i_episode)

if __name__ == "__main__":
    
    main()