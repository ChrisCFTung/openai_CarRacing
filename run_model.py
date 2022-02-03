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
state = env.reset()
lulu = lewis(state_dim=(3, 96, 96), action_dim=9, save_dir=None)
print("Please input the folder name in checkpoints")
folder = input()
lulu.load(f'checkpoints\\{folder}\\lewis_net_0.chkpt')
print("Please input the desired epsilon")
eps = input()
if float(eps) >= 0.:
    lulu.exploration_rate=float(eps)
print(lulu.exploration_rate)


while True:
    env.render()
    action = lulu.act(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    
    if done:
        break
        
env.close()