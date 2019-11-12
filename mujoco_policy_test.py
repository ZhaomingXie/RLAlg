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
#import pybullet
#import pybullet_envs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", required=True, type=str)
parser.add_argument("--env", required=True, type=str)
parser.add_argument("--learn_contact", action='store_true')
parser.add_argument("--difficulty", nargs='+', default=[])
parser.add_argument("--hidden", nargs='+', default=[256, 256])
parser.add_argument("--residual", action='store_true')
parser.add_argument("--mirror", action='store_true')
args = parser.parse_args()
import gym
env = gym.make(args.env)

if args.mirror:
	env.set_mirror(args.mirror)

num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

#model = ActorCriticNetMixtureExpert(num_inputs, num_outputs, [256, 256])
#model.load_state_dict(torch.load("expert_model/Walker2DMixExpertAug15_2019.pt"))
if args.learn_contact:
	Net = ActorCriticNetWithContact
else:
	Net = ActorCriticNet
model = Net(num_inputs, num_outputs, hidden_layer=list(map(int, args.hidden)), num_contact=2)

state_dict = torch.load(args.policy_path)
model.load_state_dict(state_dict)

if args.residual:
	base_controller = ActorCriticNet(55, num_outputs, hidden_layer=[256, 256], num_contact=2)
	base_controller.load_state_dict(torch.load("torch_model/WAlkerMocap_seed8_v3.pt"))

print(env.action_space.sample())
#model.noise = np.array([-1, -1, -1, -1, -1, -3, -2, -1, -1, -3, -2, -1, -1, -1, -1, -1, -1])
model.noise = np.ones(num_outputs)*-2.5
def run_test():
	t.sleep(1)
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
				#print(counter)
				#print(state - state2)
				state = Variable(torch.Tensor(state).unsqueeze(0))
				#print(state)
				#state = shared_obs_stats.normalize(state)
				#print(state[0, :].max)
				#index, value = (state.abs()).max(1)
				#print(index, value)
				#print("contact", state[:, -2:])
				#print("w", model.get_w(state))
				#print(model.num_contact)
				mu = model.sample_actions(state)
				print(mu)
				#mu, _ = actor(state)
				#print(model.get_actions_difference(state))

				env_action = mu.data.squeeze().numpy()
				print(env_action)
				#print(env_action[[5, 6, 9, 10]])
				state, reward, done, _ = env.step(env_action)
				#print(env.next_step_index)
				print(state)
				#env.viewer_setup()
				env.render(mode='human')
				#print(env_action)
				#print(reward)
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


def run_test_2():
	env.render()
	env.seed(0)
	env.mirror = True
	obs = env.reset()
	ep_reward = 0
	counter = 0
	step = 0
	max_reward = 1

	while True:
		with torch.no_grad():
			mu = model.sample_best_actions(torch.from_numpy(obs).float().unsqueeze(0))
			if args.residual:
				base_action = base_controller.sample_best_actions(torch.from_numpy(obs[0:55]).float().unsqueeze(0))
		
		if counter < 3:
			print(obs.mean(), mu.mean())
			counter += 1

		action = mu.squeeze().numpy()
		#print(action)
		if args.residual:
			action += base_action.squeeze().numpy()

		obs, reward, done, _ = env.step(action)
		env.render()

		step += 1
		#print(action)
		if max_reward < reward:
			print("max_reward", reward)
			max_reward = reward
		print(step, reward)
		#print(env.next_step_index)
		ep_reward += reward

		if done:
			print(step)
			#env.seed(1)
			step = 0
			print("Ep reward:", ep_reward)
			obs = env.reset()
			ep_reward = 0
			counter = 0


run_test_2()