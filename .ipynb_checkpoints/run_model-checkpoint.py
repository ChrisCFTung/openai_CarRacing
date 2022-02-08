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
from gym.wrappers import FrameStack, Monitor

from gym_torch import *
from agent import *

import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
env = env_make(skip=2, stack=3)
#env = Monitor(env, './video', force=True)
state = env.reset()
lulu = lewis(state_dim=env.observation_space.shape, action_dim=env.action_space.high[0]+1, save_dir=None)
print("Please input the folder name in checkpoints")
folder = input()
lulu.load(f'checkpoints\\{folder}\\lewis_net_9.chkpt')
print("Please input the desired epsilon")
#eps = input()
#if float(eps) >= 0.:
#    lulu.exploration_rate=float(eps)
print(lulu.exploration_rate)


while True:
    env.render()
    action = lulu.act(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    
#     done = False
#     if done:
#         print(done, info)
#         break
        
env.close()