import time as t

import statistics

import argparse
import os
import sys
import gym
from gym import wrappers
import random
import numpy as np

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

from params import Params

import pickle
from model import *#ActorCriticNet, Shared_obs_stats, ActorCriticNetMixtureExpert
import pybullet
import pybullet_envs
env = gym.make("Humanoid-v2")

num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

#model = ActorCriticNetMixtureExpert(num_inputs, num_outputs, [256, 256])
#model.load_state_dict(torch.load("expert_model/Walker2DMixExpertAug15_2019.pt"))

model = ActorCriticNetMixtureExpert(num_inputs, num_outputs, [256, 256])
model.load_state_dict(torch.load("torch_model/Humanoid_ppo_seed1.pt"))

actor = ActorNet(num_inputs, num_outputs, [256, 256])
actor.load_state_dict(torch.load("torch_model/Humanoid_TD3.pt"))

#with open('torch_model/Walker2D_2kHz_shared_obs_stats.pkl', 'rb') as input:
#	shared_obs_stats = pickle.load(input)
#with open('torch_model/cassie3dMirror2kHz_shared_obs_stats.pkl', 'rb') as input:
#	shared_obs_stats = pickle.load(input)

#print(model.robot_value_input_modules[1].weight.data)
print(env.action_space.sample())
model.noise = np.array([-1, -1, -1, -1, -1, -3, -2, -1, -1, -3, -2, -1, -1, -1, -1, -1, -1])
#model.noise = np.ones(6)
def run_test():
	t.sleep(1)
	env.render()
	state = env.reset()
	total_reward = 0
	total_10_reward = 0
	done = False
	reward_list = []

	for i in range(1):
		state = env.reset()
		total_reward = 0
		counter = 0
		while counter < 100000 and not done:
			start = t.time()
			for j in range(1):
				counter += 1
				print(counter)
				#print(state - state2)
				state = Variable(torch.Tensor(state).unsqueeze(0))
				#print(state)
				#state = shared_obs_stats.normalize(state)
				#print(state[0, :].max)
				index, value = (state.abs()).max(1)
				print(index, value)
				mu = model.sample_best_actions(state)
				mu, _ = actor(state)
				#print(model.get_actions_difference(state))

				env_action = mu.data.squeeze().numpy()
				#print(env_action[[5, 6, 9, 10]])
				state, reward, done, _ = env.step(env_action)
				#print(reward)
				env.viewer_setup()
				env.render(mode='human')
				print(env_action)
				print(reward)
				total_reward += reward
				force = np.zeros(12)
				pos = np.zeros(6)
			while True:
				stop = t.time()
				#print(stop-start)
				if stop - start > 0.02:
					break
				#print("stop")
			#total_reward += reward
		done = False
		counter = 0
		reward_list.append(total_reward)
		total_10_reward += total_reward
		print("total rewards", total_reward)
	print(total_10_reward)
	print(statistics.mean(reward_list))
	print(statistics.stdev(reward_list))

run_test()