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

# reduce the action space with a wrapper
class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.__actions__ = [(-1, 0.2, 0.), (0, 0.2, 0.), (1, 0.2, 0.),
                            (-1, 0.5, 0.), (0, 0.5, 0.), (1, 0.5, 0.), #           Action Space Structure
                            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
                            (-1, 0, 0.8), (0, 0, 0.8), (1, 0, 0.8), # Range        -1~1       0~1   0~1
                            (-1, 0, 0.3), (0, 0, 0.3), (1, 0, 0.3),
                            (-1, 0,   0), (0, 0,   0), (1, 0,   0)]
#                            [(-1.0, 0.0, 0.2), (0.0, 1.0, 0.0), (1.0, 0.0, 0.2), #           Action Space Structure
#                             (-0.5, 0.5, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0), #        (Steering Wheel, Gas, Break)
#                             (-0.5, 0.0, 0.5), (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), # Range        -1~1       0~1   0~1
#                             ( 0.0, 0.2, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
#         [
#             (-1.,0.,0.),
#             (-1.,0.,0.),
#             (0.,1.,0.),
#             (0.,0.,1.),
#             (0.,0.,0.),
#         ]
        
        self.action_space = Box(0, len(self.__actions__)-1, shape=(1,), dtype=np.int8)
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(self.__actions__[action])
        # modify ...
        return next_state, reward, done, info

# Skip some frames
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

#make the track into gray scale, and also cut off the bottom
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = self.observation_space.shape[:2]
        obs_shape = (h-16, w)   # remove the bottom status bar
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2,0,1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation
    
    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        observation = observation.squeeze(0)
        return observation[:-16]
    
import time, datetime

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
    
def env_make(skip=2, stack=3):
    env = gym.make('CarRacing-v0')
    env.unwrapped.verbose=0
    env = SkipFrame(env, 2)
    env = BasicWrapper(env)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=3)
    return env
