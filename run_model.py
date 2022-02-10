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

import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', dest='chkdir', type=str, action='store',
                    help='directory name of the checkpoint')
parser.add_argument('--id', dest='chkid', type=str, action='store',
                    help='id of the checkpoint version')
parser.add_argument('--eps', dest='epsilon', type=float, action='store',
                    default = 0.1,
                    help='value of the epsilon for the agent')
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK']='True'
env = env_make(skip=2, stack=3)
#env = Monitor(env, './video', force=True)
state = env.reset()
lulu = lewis(state_dim=env.observation_space.shape, action_dim=env.action_space.high[0]+1, save_dir=None)
folder = args.chkdir
idx = args.chkid
lulu.load(f'checkpoints\\{folder}\\lewis_net_{idx}.chkpt')
eps = args.epsilon
if float(eps) >= 0.:
   lulu.exploration_rate=float(eps)
print(lulu.exploration_rate)


for frame in range(2000):
    env.render()
    action = lulu.act(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    # if done:
    #     print(done, info)
    #     break
env.close()