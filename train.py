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

from gym_torch import *
from agent import *

import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = env_make()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

lulu = lewis(state_dim=env.observation_space.shape, action_dim=env.action_space.high[0]+1, save_dir=save_dir)
lulu.load('checkpoints\\2022-03-11T10-43-24\\lewis_net_2.chkpt')
lulu.exploration_rate_min = 0.1
lulu.exploration_rate = 0.1

logger = MetricLogger(save_dir)

episodes = 2000
for ep in range(episodes):
    #env.seed(1)
    state = env.reset()
    
    while True:
        action = lulu.act(state)
        
        next_state, reward, done, info = env.step(action)
        
        neg_counter = 0 
        if reward < 0 and logger.curr_ep_length > 100:
            neg_counter += 1
        else:
            neg_counter = 0
        
        lulu.cache(state, next_state, action, reward, done)
        q, loss = lulu.learn()
        logger.log_step(reward, loss, q)
        state = next_state
        
        if done or logger.curr_ep_reward<0 or neg_counter>20:
            break
    
    logger.log_episode()
    
    if ep%20 ==0:
        logger.record(episode=ep, epsilon=lulu.exploration_rate, step=lulu.curr_step)
        
lulu.save()        
env.close()
