import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import copy, random

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

class LewisNet(nn.Module):
    """
    cnn structure
    input -> (conv2d + relu) X 3 -> flatten -> (dense + relu) X 2 -> output
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        if h != 80:
            raise ValueError(f"Expecting input height: 80, got: {h}")
        if w != 96:
            raise ValueError(f"Expecting input height: 96, got: {w}") 
            
        # Create Q
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
        # Create Q_target
        self.target = copy.deepcopy(self.online)
        # Freeze Q_target
        for p in self.target.parameters():
            p.requires_grad = False
            
    def forward(self, inp, model):
        if model == "online":
            return self.online(inp)
        elif model == "target":
            return self.target(inp)
        

class lewis:
    def __init__(self, state_dim, action_dim, save_dir):
        """
        An agent to drive the car
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        
        self.use_cuda = torch.cuda.is_available()
        self.net = LewisNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
            
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.3
        self.curr_step = 0
        
        self.save_every = 1e5 # no. of experiences between saving
        
        # cache and recall
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # learn
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_fn = torch.nn.MSELoss() #torch.nn.SmoothL1Loss()
        
        self.burnin = 1000
        self.learn_every = 3
        self.sync_every = 200
        
    
    def act(self, state):
        """
        Given a state choose an epsilon-greedy action
        
        Inputs:
        state(LazyFrame): a single observation, dimension = state_dim
        Outputs:
        action_idx(int): an integer representing the action in the action space
        """
        # Explore
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
            
        # Exploit
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()
            
        # Decrease exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        # increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """
        Add the experience to memory
        Inputs:
        state (LazyFrame)
        next_state (LazyFrame)
        action (int)
        reward (float)
        done (bool)
        
        """
        state = state.__array__()
        next_state = next_state.__array__()
        
        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])
            
        self.memory.append((state, next_state, action, reward, done,))
            
    def recall(self):
        """Sample experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def learn(self):
        """Update the Q function with a batch of experience"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
            
        if self.curr_step % self.save_every == 0:
            self.save()
            
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        state, next_state, action, reward, done = self.recall()
        
        td_est = self.td_estimate(state, action)
        
        td_tgt = self.td_target(reward, next_state, done)
        
        loss = self.update_Q_online(td_est, td_tgt)
        
        return (td_est.mean().item(), loss)
    
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma*next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        # for tar_p, p in zip(self.net.target.parameters(), self.net.online.parameters()):
        #     tar_p.data.mul_(0.95)
        #     tar_p.data.add_((1-0.95)*p.data)
        self.net.target.load_state_dict(self.net.online.state_dict())
        
    def save(self):
        save_path = (self.save_dir / f"lewis_net_{int(self.curr_step//self.save_every)}.chkpt")
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path,)
        print(f"LewisNet saved to {save_path} at step {self.curr_step}")
        
    def load(self, path_to_chkpt):
        chkpt = torch.load(path_to_chkpt)
        self.net.load_state_dict(chkpt['model'])
        self.exploration_rate = chkpt['exploration_rate']
        print('Checkpoint loaded')
