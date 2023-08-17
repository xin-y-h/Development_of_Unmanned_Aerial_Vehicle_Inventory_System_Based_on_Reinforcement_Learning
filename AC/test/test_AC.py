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

from env import env
from network import Actor

from tensorboardX import SummaryWriter
from datetime import datetime
# tensorboard --logdir=runs

# outputfile='./outputpicture/'#outfile path

time = datetime.now()
path = f'runs/{time.strftime("%Y_%m_%d_%H_%M_%S"+"ppo_ok")}'
writer = SummaryWriter(path)

random.seed()#每次random都不一樣(給參數就會變成每次random給的數都一樣)
n_epochs=24
batch_size=3
# nvidia-smi
lr=0.0003
lr_decay=0.9
gamma=0.9
device="cuda"
n_episodes=3000
# len_memory=2000
len_memory=20



# 暫存資料
class MemoryBuffer():
    def __init__(self, batch_size):
        # self.states = []
        self.states_other = []
        self.states_img = []
        
        self.actions = []
        self.rewards = []
        self.dones = []
        self.states_img_next=[]
        self.states_other_next=[]

        self.batch_size = batch_size

    def store(self,states_other,states_img, action, reward, done, states_other_next,states_img_next):
        
        self.states_other.append(states_other)
        self.states_img.append(states_img)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.states_other_next.append(states_other_next)
        self.states_img_next.append(states_img_next)
        print("store")

    def clear(self):
       
        self.states_other = []
        self.states_img = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.states_other_next=[]
        self.states_img_next=[]
        print("clear")



        
    def generate_batch(self):
        n_states = len(self.states_img)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        
        return np.array(list(self.states_other), dtype=np.float32),np.array(list(self.states_img), dtype=np.float32), np.array(list(self.actions), dtype=np.float32), np.array(list(self.rewards), dtype=np.float32),\
               np.array(list(self.dones), dtype=np.float32),np.array(list(self.states_other_next), dtype=np.float32),np.array(list(self.states_img_next), dtype=np.float32),batches

        
    def __len__(self):
        return len(self.states_img)


# agent
class Actor_Net():
    def __init__(self,n_epochs, batch_size, lr,lr_decay, gamma, device):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.decay_rate = lr_decay
        self.device = torch.device(device)
        #self.model_dir = model_dir #dir資料路徑

        self.Actor = Actor().to(self.device)
        
        self.optimizer = optim.Adam(self.Actor.parameters(), lr, weight_decay=0.0001)
       

        self.memory = MemoryBuffer(self.batch_size)

    def remember(self, states_other,states_img, action, reward, done, states_other_next,states_img_next):
        self.memory.store(states_other,states_img, action, reward, done, states_other_next,states_img_next)
    
    def choose_action(self, obs,action_to_angle):
        obs2=torch.tensor(obs[4]).to(self.device)
        
        action_pro, critic_value = self.Actor(obs2)
        
        action_dist=torch.distributions.Categorical(action_pro)#分佈曲線
        print(action_dist)
        action = action_dist.sample()#以分佈曲線取樣

        
        print(type(action))
        print("action_action",action)

        angle_change=random.randrange(0,8,1)
        move_change=random.randrange(0,3,1)
        move_change_2=random.randrange(0,5,1)


        if (not action_to_angle):
            
            if(obs[0]==35 and (action==0 or action==4 or action==6)):

                if(obs[1]==0):
                    if move_change==0:
                        action=torch.tensor([2])
                    if move_change==1:
                        action=torch.tensor([3])
                    if move_change==2:
                        action=torch.tensor([7])

                elif(obs[1]==-2):
                    if move_change==0:
                        action=torch.tensor([1])
                    if move_change==1:
                        action=torch.tensor([2])
                    if move_change==2:
                        action=torch.tensor([5])
                
                else:
                    if move_change_2==0:
                        action=torch.tensor([1])
                    if move_change_2==1:
                        action=torch.tensor([2])
                    if move_change_2==2:
                        action=torch.tensor([3])
                    if move_change_2==3:
                        action=torch.tensor([5])
                    if move_change_2==4:
                        action=torch.tensor([7])
                
            elif(obs[0]==2  and (action==2 or action==5 or action==7)):
                
                if(obs[1]==0):
                    if move_change==0:
                        action=torch.tensor([0])
                    if move_change==1:
                        action=torch.tensor([3])
                    if move_change==2:
                        action=torch.tensor([6])

                elif(obs[1]==-2):
                    if move_change==0:
                        action=torch.tensor([0])
                    if move_change==1:
                        action=torch.tensor([1])
                    if move_change==2:
                        action=torch.tensor([4])
                else:
                    if move_change_2==0:
                        action=torch.tensor([0])
                    if move_change_2==1:
                        action=torch.tensor([1])
                    if move_change_2==2:
                        action=torch.tensor([3])
                    if move_change_2==3:
                        action=torch.tensor([4])
                    if move_change_2==4:
                        action=torch.tensor([6])

            elif(obs[1]==0 and (action==1 or action==4 or action==5)):
                if( obs[0]==35):
                    if move_change==0:
                        action=torch.tensor([2])
                    if move_change==1:
                        action=torch.tensor([3])
                    if move_change==2:
                        action=torch.tensor([7])

                elif(obs[0]==2):
                    if move_change==0:
                        action=torch.tensor([0])
                    if move_change==1:
                        action=torch.tensor([3])
                    if move_change==2:
                        action=torch.tensor([6])
                else:
                    if move_change_2==0:
                        action=torch.tensor([0])
                    if move_change_2==1:
                        action=torch.tensor([2])
                    if move_change_2==2:
                        action=torch.tensor([3])
                    if move_change_2==3:
                        action=torch.tensor([6])
                    if move_change_2==4:
                        action=torch.tensor([7])
                
            elif(obs[1]==-2 and (action==3 or action==6 or action==7)):
                if( obs[0]==35):
                    if move_change==0:
                        action=torch.tensor([1])
                    if move_change==1:
                        action=torch.tensor([2])
                    if move_change==2:
                        action=torch.tensor([5])
                elif(obs[0]==2):
                    if move_change==0:
                        action=torch.tensor([0])
                    if move_change==1:
                        action=torch.tensor([1])
                    if move_change==2:
                        action=torch.tensor([4])
                else:
                    if move_change_2==0:
                        action=torch.tensor([0])
                    if move_change_2==1:
                        action=torch.tensor([1])
                    if move_change_2==2:
                        action=torch.tensor([2])
                    if move_change_2==3:
                        action=torch.tensor([4])
                    if move_change_2==4:
                        action=torch.tensor([5])

        else:
            action=random.randrange(8,17,1)
            action=torch.tensor([action])

            if(obs[2]==-40 and action==8):
                if angle_change==0:
                    action=torch.tensor([9])
                if angle_change==1:
                    action=torch.tensor([10])
                if angle_change==2:
                    action=torch.tensor([11])
                if angle_change==3:
                    action=torch.tensor([12])
                if angle_change==4:
                    action=torch.tensor([13])
                if angle_change==5:
                    action=torch.tensor([14])
                if angle_change==6:
                    action=torch.tensor([15])
                if angle_change==7:
                    action=torch.tensor([16])
                
                # input()
            
            elif(obs[2]==-30 and action==9):
                if angle_change==0:
                    action=torch.tensor([8])
                if angle_change==1:
                    action=torch.tensor([10])
                if angle_change==2:
                    action=torch.tensor([11])
                if angle_change==3:
                    action=torch.tensor([12])
                if angle_change==4:
                    action=torch.tensor([13])
                if angle_change==5:
                    action=torch.tensor([14])
                if angle_change==6:
                    action=torch.tensor([15])
                if angle_change==7:
                    action=torch.tensor([16])
                # input()

            elif(obs[2]==-20 and action==10):
                if angle_change==0:
                    action=torch.tensor([8])
                if angle_change==1:
                    action=torch.tensor([9])
                if angle_change==2:
                    action=torch.tensor([11])
                if angle_change==3:
                    action=torch.tensor([12])
                if angle_change==4:
                    action=torch.tensor([13])
                if angle_change==5:
                    action=torch.tensor([14])
                if angle_change==6:
                    action=torch.tensor([15])
                if angle_change==7:
                    action=torch.tensor([16])
                # input()
            
            elif(obs[2]==-10 and action==11):
                if angle_change==0:
                    action=torch.tensor([8])
                if angle_change==1:
                    action=torch.tensor([9])
                if angle_change==2:
                    action=torch.tensor([10])
                if angle_change==3:
                    action=torch.tensor([12])
                if angle_change==4:
                    action=torch.tensor([13])
                if angle_change==5:
                    action=torch.tensor([14])
                if angle_change==6:
                    action=torch.tensor([15])
                if angle_change==7:
                    action=torch.tensor([16])
                # input()
            
            elif(obs[2]==0 and action==12):
                if angle_change==0:
                    action=torch.tensor([8])
                if angle_change==1:
                    action=torch.tensor([9])
                if angle_change==2:
                    action=torch.tensor([10])
                if angle_change==3:
                    action=torch.tensor([11])
                if angle_change==4:
                    action=torch.tensor([13])
                if angle_change==5:
                    action=torch.tensor([14])
                if angle_change==6:
                    action=torch.tensor([15])
                if angle_change==7:
                    action=torch.tensor([16])
            
            elif(obs[2]==10 and action==13):
                if angle_change==0:
                    action=torch.tensor([8])
                if angle_change==1:
                    action=torch.tensor([9])
                if angle_change==2:
                    action=torch.tensor([10])
                if angle_change==3:
                    action=torch.tensor([11])
                if angle_change==4:
                    action=torch.tensor([12])
                if angle_change==5:
                    action=torch.tensor([14])
                if angle_change==6:
                    action=torch.tensor([15])
                if angle_change==7:
                    action=torch.tensor([16])
            
            elif(obs[2]==20 and action==14):
                if angle_change==0:
                    action=torch.tensor([8])
                if angle_change==1:
                    action=torch.tensor([9])
                if angle_change==2:
                    action=torch.tensor([10])
                if angle_change==3:
                    action=torch.tensor([11])
                if angle_change==4:
                    action=torch.tensor([12])
                if angle_change==5:
                    action=torch.tensor([13])
                if angle_change==6:
                    action=torch.tensor([15])
                if angle_change==7:
                    action=torch.tensor([16])

            elif(obs[2]==30 and action==15):
                if angle_change==0:
                    action=torch.tensor([8])
                if angle_change==1:
                    action=torch.tensor([9])
                if angle_change==2:
                    action=torch.tensor([10])
                if angle_change==3:
                    action=torch.tensor([11])
                if angle_change==4:
                    action=torch.tensor([12])
                if angle_change==5:
                    action=torch.tensor([13])
                if angle_change==6:
                    action=torch.tensor([14])
                if angle_change==7:
                    action=torch.tensor([16])

            elif(obs[2]==40 and action==16):
                if angle_change==0:
                    action=torch.tensor([8])
                if angle_change==1:
                    action=torch.tensor([9])
                if angle_change==2:
                    action=torch.tensor([10])
                if angle_change==3:
                    action=torch.tensor([11])
                if angle_change==4:
                    action=torch.tensor([12])
                if angle_change==5:
                    action=torch.tensor([13])
                if angle_change==6:
                    action=torch.tensor([14])
                if angle_change==7:
                    action=torch.tensor([15])
                # input()
    
        action=torch.tensor(action).to(device)
        
        prob = action_dist.log_prob(action)

        print("action",action)
        # input()
        
        print("chose_action")
        

        return action.detach(), prob.detach(), critic_value.detach()
        

    def lr_decay(self):
        self.optimizer.param_groups[0]["lr"] *= self.decay_rate #呼叫時每次都*一個 <1 的數來下降


    def learn(self):
        print("Learning Policy ...")
        for _ in range(self.n_epochs):
            
            _,states_img_np,_, rewards_np, dones_np,_,states_img_next_np,batches = self.memory.generate_batch()
            
            
            history_total_loss = []
            history_actor_loss = []
            history_critic_loss = []
            
            for batch in batches:
                
                states_img = torch.tensor(states_img_np[batch]).to(self.device)
                states_img_next = torch.tensor(states_img_next_np[batch]).to(self.device)
                rewards=torch.tensor(rewards_np[batch]).to(self.device)
                done=torch.tensor(dones_np[batch]).to(self.device)
                action_pro, critic_value = self.Actor(states_img)                
                _, critic_value_next = self.Actor(states_img_next)

                action_dist=torch.distributions.Categorical(action_pro)#分佈曲線
                action = action_dist.sample()#以分佈曲線取樣
                prob = action_dist.log_prob(action)
                actor_loss=-prob*rewards                

                for i in range(len(batch)):
                    
                    if done[i]:
                        critic_loss = rewards_np[i]- critic_value[i]
                    else:
                        critic_loss = rewards_np[i] + self.gamma*critic_value_next[i] - critic_value[i]

                    total_loss = actor_loss[i] + 0.5*critic_loss

                    history_actor_loss.append(actor_loss[i].cpu().item())
                    history_total_loss.append(total_loss.cpu().item())
                    history_critic_loss.append(critic_loss.cpu().item())


                for i in range(len(history_total_loss)):
                    writer.add_scalar('actor_loss', history_actor_loss[i],i)
                    writer.add_scalar('critic_loss',history_critic_loss[i],i)
                    writer.add_scalar('total_loss', history_total_loss[i],i)

                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer.step()

            
        self.memory.clear()
       
    
    


def main():
    
    train_env=env()
    print("Create an Environment ...")
    agent = Actor_Net(n_epochs, batch_size, lr, lr_decay, gamma, device)
    print("Create an Agent ...")  
   

    count_lr=0
    all_ep_r=[]
    # all_reward_totle=[]
    for i_episode in range(n_episodes):   
        obs=train_env.reset()
        action_to_angle=obs[5]
        
        reward_totle = 0 
        
        while not train_env.done:
            print("train_env",i_episode)
            action,_, _ = agent.choose_action(obs,action_to_angle)
            print("i_episode_action",action)
            next_obs, reward, done,action_to_angle = train_env.env_step(obs[0],obs[1],obs[2],action,obs[3])
            reward_totle += reward 
            agent.remember(torch.tensor(obs[0:4]).tolist(),(obs[4].squeeze(0)).tolist(), action.tolist(), reward, done, torch.tensor(next_obs[0:4]).tolist(),(next_obs[4].squeeze(0)).tolist())
            
            
            if len(agent.memory) >= len_memory and done:
                agent.learn()
                count_lr+=1
                print("len")
            obs = next_obs

            if done:
                print("done ...",done)  
                break
        print("count_lr .................................",count_lr) 
        print("lr: ", agent.optimizer.param_groups[0]["lr"])

        # 固定次數更改lr（減少幫助收斂）
        if((count_lr % 100==0) and (count_lr != 0)):
            agent.lr_decay()
            
        if i_episode == 0:
                all_ep_r.append(reward_totle)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + reward_totle * 0.1)

        writer.add_scalar('episode reward', reward_totle,i_episode)

        writer.add_scalar('averaged episode reward', all_ep_r[i_episode],i_episode)

        print("\rEp: {} |rewards: {}".format(i_episode, reward_totle), end="")
        print("all_ep_r",all_ep_r[i_episode])
    

        # 保存參數
        if i_episode % 1 == 0 and i_episode >= 20:
            save_data1 = {'net': agent.Actor.state_dict(), 'opt': agent.optimizer.state_dict(), 'i': i_episode}
            
            if(i_episode==20 or all_ep_r[i_episode]>all_ep_r[i_episode-20]):
            
                torch.save(save_data1, "./weight/AC_model_actor.pth")
        
        save_data1 = {'net': agent.Actor.state_dict(), 'opt': agent.optimizer.state_dict(), 'i': i_episode}
       
        torch.save(save_data1, "./weight/AC_model_actor_last.pth")
  

if __name__ == "__main__":
    main()
