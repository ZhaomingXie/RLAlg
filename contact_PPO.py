import argparse
import os
import sys
import random
import numpy as np
import scipy

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

from params import Params

import pickle
import time as t

from model import ActorCriticNet, Shared_obs_stats, ActorCriticNetWithContact

import statistics
import matplotlib.pyplot as plt
from operator import add, sub

import pickle
import threading
import torch.multiprocessing as mp
import queue
from utils import TrafficLight
from utils import Counter
from radam import RAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# try:
#     mp.set_start_method('spawn')
# except RuntimeError:
#     pass
import sys
sys.path.append('/home/zhaoming/Documents/dev/gym/gym/envs/mujoco')

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.sample_index = 0

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

    def push_half(self, events):
        temp_memory = []
        for event in zip(*events):
            temp_memory.append(event)
        self.memory = self.memory + temp_memory[len(temp_memory)//4:3*len(temp_memory)//2]

        while len(self.memory)>self.capacity:
            del self.memory[0]

    def push_half(self, events):
        temp_memory = []
        for event in zip(*events):
            temp_memory.append(event)
        self.memory = self.memory + temp_memory[2*len(temp_memory)//4:len(temp_memory)]

        while len(self.memory)>self.capacity:
            del self.memory[0]

    def clear(self):
        self.memory = []
        self.sample_index = 0

    def sample(self, batch_size):
        #print(len(self.memory), batch_size)
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: np.concatenate(x, 0), samples)

    def clean_memory(self):
        while len(self.memory) > self.capacity:
            del self.memory[0]

    def shuffle(self):
        random.shuffle(self.memory)

    def sample_one_at_a_time(self):
        samples = zip(*self.memory[self.sample_index:self.sample_index+1])
        self.sample_index += 1
        return map(lambda x: np.concatenate(x, 0), samples)

def normal(x, mu, log_std):
    a = (x - mu)/(log_std.exp())
    a = -0.5 * a.pow(2)
    a = torch.sum(a, dim=1)
    b = torch.sum(log_std, dim=1)
    #print(a-b)
    return a-b

class RL(object):
    def __init__(self, env, hidden_layer=[64, 64], contact=False):
        self.env = env
        #self.env.env.disableViewer = False
        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        self.hidden_layer = hidden_layer
        self.num_contact = 2

        self.params = Params()
        if contact:
            self.Net = ActorCriticNetWithContact
        else:
            self.Net = ActorCriticNet
        self.model = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer, num_contact=self.num_contact)
        self.model.share_memory()
        self.shared_obs_stats = Shared_obs_stats(self.num_inputs)
        self.memory = ReplayMemory(10000000)
        self.value_memory = ReplayMemory(10000000)
        self.test_mean = []
        self.test_std = []

        self.noisy_test_mean = []
        self.noisy_test_std = []
        self.fig = plt.figure()
        #self.fig2 = plt.figure()
        self.lr = self.params.lr
        plt.show(block=False)

        self.test_list = []
        self.noisy_test_list = []
        self.queue = mp.Queue()
        self.value_queue = mp.Queue()

        self.mpdone = [mp.Event(), mp.Event(), mp.Event(), mp.Event()]

        self.process = []
        self.traffic_light = TrafficLight()
        self.counter = Counter()

        self.best_trajectory = ReplayMemory(5000)
        self.best_score_queue = mp.Queue()
        self.best_score = mp.Value("f", 0)
        self.max_reward = mp.Value("f", 1)

        self.expert_trajectory = ReplayMemory(1e7)

        self.validation_trajectory = ReplayMemory(6000*9)

        self.best_validation = 1.0
        self.current_best_validation = 1.0

        self.return_obs_stats = Shared_obs_stats(1)

        self.gpu_model = self.Net(self.num_inputs, self.num_outputs,self.hidden_layer, num_contact=self.num_contact)

        self.base_controller = None

    def normalize_data(self, num_iter=1000, file='shared_obs_stats.pkl'):
        state = self.env.reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        #model_old = ActorCriticNet(self.num_inputs, self.num_outputs,self.hidden_layer)
        #model_old.load_state_dict(self.model.state_dict())
        for i in range(num_iter):
            print(i)
            self.shared_obs_stats.observes(state)
            state = self.shared_obs_stats.normalize(state)#.to(device)
            #mu = self.model.sample_actions(state)
            #action = mu#(mu + log_std.exp()*Variable(eps))
            #env_action = action.cpu().data.squeeze().numpy()
            env_action = np.random.randn(self.num_outputs)
            state, reward, done, _ = self.env.step(env_action*0)

            if done:
                state = self.env.reset()

            state = Variable(torch.Tensor(state).unsqueeze(0))

        with open(file, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def run_test(self, num_test=1):
        state = self.env.reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        ave_test_reward = 0

        total_rewards = []
        
        for i in range(num_test):
            total_reward = 0
            while True:
                state = self.shared_obs_stats.normalize(state)
                mu = self.model.sample_best_actions(state)
                action = mu.cpu().data.squeeze().numpy()
                if self.base_controller is not None:
                    base_action = self.base_controller.sample_best_actions(state)
                    action += base_action.cpu().data.squeeze().numpy()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                #print(state)
                #print("done", done, "state", state)

                if done:
                    state = self.env.reset()
                    #print(self.env.position)
                    #print(self.env.time)
                    state = Variable(torch.Tensor(state).unsqueeze(0))
                    ave_test_reward += total_reward / num_test
                    total_rewards.append(total_reward)
                    break
                state = Variable(torch.Tensor(state).unsqueeze(0))
        #print("avg test reward is", ave_test_reward)

        reward_mean = statistics.mean(total_rewards)
        reward_std = statistics.stdev(total_rewards)
        self.test_mean.append(reward_mean)
        self.test_std.append(reward_std)
        self.test_list.append((reward_mean, reward_std))
        #print(self.model.state_dict())

    def run_test_with_noise(self, num_test=10):
        state = self.env.reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        ave_test_reward = 0

        total_rewards = []
        
        for i in range(num_test):
            total_reward = 0
            while True:
                state = self.shared_obs_stats.normalize(state)
                mu = self.model.sample_actions(state)
                eps = torch.randn(mu.size())
                action = (mu + 0.0*Variable(eps))
                action = action.cpu().data.squeeze().numpy()
                if self.base_controller is not None:
                    base_action = self.base_controller.sample_best_actions(state)
                    action += base_action.cpu().data.squeeze().numpy()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    state = self.env.reset()
                    state = Variable(torch.Tensor(state).unsqueeze(0))
                    ave_test_reward += total_reward / num_test
                    total_rewards.append(total_reward)
                    break
                state = Variable(torch.Tensor(state).unsqueeze(0))
        #print("avg test reward is", ave_test_reward)

        reward_mean = statistics.mean(total_rewards)
        reward_std = statistics.stdev(total_rewards)
        self.noisy_test_mean.append(reward_mean)
        self.noisy_test_std.append(reward_std)
        self.noisy_test_list.append((reward_mean, reward_std))

    def plot_statistics(self):
        
        ax = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        low = []
        high = []
        index = []
        noisy_low = []
        noisy_high = []
        for i in range(len(self.test_mean)):
            low.append(self.test_mean[i] - self.test_std[i])
            high.append(self.test_mean[i] + self.test_std[i])
            noisy_low.append(self.noisy_test_mean[i]-self.noisy_test_std[i])
            noisy_high.append(self.noisy_test_mean[i]+self.noisy_test_std[i])
            index.append(i)
        plt.xlabel('iterations')
        plt.ylabel('average rewards')
        ax.plot(self.test_mean, 'b')
        ax2.plot(self.noisy_test_mean, 'g')
        ax.fill_between(index, low, high, color='cyan')
        ax2.fill_between(index, noisy_low, noisy_high, color='r')
        #ax.plot(map(sub, test_mean, test_std))
        self.fig.canvas.draw()

    def collect_samples(self, num_samples, start_state=None, noise=-2.0, env_index=0, random_seed=1):

        random.seed(random_seed)
        torch.manual_seed(random_seed+1)
        np.random.seed(random_seed+2)
        env.seed(random_seed + 3)
        #env.seed(random_seed+3)
        #print(noise)

        if start_state == None:
            start_state = self.env.reset()
        samples = 0
        done = False
        states = []
        next_states = []
        actions = []
        rewards = []
        values = []
        q_values = []
        real_rewards = []
        log_probs = []
        noise = self.base_noise * self.explore_noise.value
        self.model.set_noise(noise)

        state = start_state
        state = Variable(torch.Tensor(state).unsqueeze(0))
        total_reward = 0
        #q_value = Variable(torch.zeros(1, 1))
        while True:
            noise = self.base_noise * self.explore_noise.value
            self.model.set_noise(noise)
            #print("local", self.model.p_fcs[1].bias.data[0])
            #self.model.load_state_dict(torch.load(self.model_name))
            signal_init = self.traffic_light.get()
            score = 0
            while samples < num_samples and not done:
                #self.shared_obs_stats.observes(state)

                states.append(state.cpu().data.numpy())
                #self.shared_obs_stats.observes(state)
                #print("samples", samples)
                state = self.shared_obs_stats.normalize(state)
                action = self.model.sample_actions(state)
                log_prob = self.model.calculate_prob(state, action)

                actions.append(action.cpu().data.numpy())
                log_probs.append(log_prob.data.numpy())
               
                env_action = action.data.squeeze().numpy()
                if self.base_controller is not None:
                    base_action = self.base_controller.sample_best_actions(state)
                    env_action += base_action.cpu().data.squeeze().numpy()
                state, reward, done, _ = self.env.step(env_action)
                score += reward
                if reward > self.max_reward.value:
                    self.max_reward.value = reward
                if self.max_reward.value > 50:
                    self.max_reward.value = 50
                #print(self.max_reward.value)
                #reward *= 0.3
                rewards.append(Variable(reward * torch.ones(1)).data.numpy())
                
                real_rewards.append(Variable(reward * torch.ones(1)).data.numpy())
                
                state = Variable(torch.Tensor(state).unsqueeze(0))
                next_states.append(state.cpu().data.numpy())
                next_state = self.shared_obs_stats.normalize(state)
                
                samples += 1
  

            state = self.shared_obs_stats.normalize(state)

            v = (self.model.get_value(state))*self.max_reward.value# / self.return_obs_stats.std) + self.return_obs_stats.mean
            if self.base_controller is not None:
                v += self.base_controller.get_value(state)*self.max_reward.value
            if done:
                R = torch.zeros(1, 1)
            else:
                R = v.data
            R = Variable(R)
            for i in reversed(range(len(real_rewards))):
                reward = Variable(torch.from_numpy(real_rewards[i]).unsqueeze(0))
                R = self.params.gamma * R + reward#self.return_obs_stats.normalize(reward)# Variable(torch.from_numpy(real_rewards[i]))
                q_values.insert(0, R.cpu().data.numpy())
                #self.return_obs_stats.observes(R)

            #mirror
            # mirror_states = np.array(states)
            # mirror_actions = np.array(actions)
            # (
            #     negation_obs_indices,
            #     right_obs_indices,
            #     left_obs_indices,
            #     negation_action_indices,
            #     right_action_indices,
            #     left_action_indices,
            # ) = self.env.get_mirror_indices()
            
            # mirror_states[:, :, negation_obs_indices] *= -1
            # rl = np.concatenate((right_obs_indices, left_obs_indices))
            # lr = np.concatenate((left_obs_indices, right_obs_indices))
            # mirror_states[:, :, rl] = mirror_states[:, :,lr]         

            # #mirror_actions = self.model.sample_best_actions(batch_states)
            # mirror_actions[:, :, negation_action_indices] = mirror_actions[:, :, negation_action_indices] * -1
            # rl = np.concatenate((right_action_indices, left_action_indices))
            # lr = np.concatenate((left_action_indices, right_action_indices))
            # mirror_actions[:, :, rl] = mirror_actions[:, :, lr]
            # mirror_states = list(mirror_states)
            # mirror_actions = list(mirror_actions)
            # #self.queue.put([mirror_states, mirror_actions, np.copy(next_states), np.copy(rewards), np.copy(q_values), np.copy(log_probs)])

            # value_states = states + mirror_states
            # value_actions = actions + mirror_actions
            # value_next_states = next_states + next_states
            # value_rewards = rewards + rewards
            # value_q_values = q_values + q_values
            # value_log_probs = log_probs + log_probs
            self.queue.put([states, actions, next_states, rewards, q_values, log_probs])
            #self.value_queue.put([value_states, value_actions, value_next_states, value_rewards, value_q_values, value_log_probs])
            self.counter.increment()
            self.env.reset()
           
            while self.traffic_light.get() == signal_init:
                pass
            start_state = self.env.reset()
            state = start_state
            state = Variable(torch.Tensor(state).unsqueeze(0))
            total_reward = 0
            samples = 0
            done = False
            states = []
            next_states = []
            actions = []
            rewards = []
            values = []
            q_values = []
            real_rewards = []
            log_probs = []
            #print("child", self.model.noise)
            #if self.model.noise[0] > -2:
            #    self.model.noise *= 1.001

    def collect_expert_samples(self, num_samples, filename, noise=-2.0,validation=False, difficulty=[0, 0]):
        import gym
        expert_env = gym.make("mocca_envs:Walker3DStepperEnv-v0")
        expert_env.set_difficulty(difficulty)
        start_state = expert_env.reset()
        samples = 0
        done = False
        states = []
        next_states = []
        actions = []
        rewards = []
        q_values = []
        model_expert = self.Net(self.num_inputs, self.num_outputs,self.hidden_layer)

        model_expert.load_state_dict(torch.load(filename))
        policy_noise = noise * np.ones(self.num_outputs)
        model_expert.set_noise(policy_noise)

        state = start_state
        state = Variable(torch.Tensor(state).unsqueeze(0))
        total_reward = 0
        total_sample = 0
        #q_value = Variable(torch.zeros(1, 1))
        if validation:
            max_sample = 300
        else:
            max_sample = 50000
        while total_sample < max_sample:
            score = 0
            while samples < num_samples and not done:
                state = self.shared_obs_stats.normalize(state)

                states.append(state.data.numpy())
                mu = model_expert.sample_best_actions(state)
                actions.append(mu.data.numpy())
                eps = torch.randn(mu.size())
                if validation:
                    weight = 0.1
                else:
                    weight = 0.1
                env_action = model_expert.sample_actions(state)
                env_action = env_action.data.squeeze().numpy()
                
                state, reward, done, _ = expert_env.step(env_action)
                reward = 1
                rewards.append(Variable(reward * torch.ones(1)).data.numpy())
                state = Variable(torch.Tensor(state).unsqueeze(0))

                next_state = self.shared_obs_stats.normalize(state)
                next_states.append(next_state.data.numpy())

                samples += 1
                #total_sample += 1
                score += reward
            print("expert score", score)
  

            state = self.shared_obs_stats.normalize(state)
            #print(state)
            v = model_expert.get_value(state)
            if done:
                R = torch.zeros(1, 1)
            else:
                R = v.data
                R = torch.ones(1, 1) * 100
            R = Variable(R)
            for i in reversed(range(len(rewards))):
                R = self.params.gamma * R + Variable(torch.from_numpy(rewards[i]))
                q_values.insert(0, R.data.numpy())
            
            if not validation and score >= num_samples:
                self.expert_trajectory.push([states, actions, next_states, rewards, q_values])
                total_sample += num_samples
            elif score >= num_samples:
                self.validation_trajectory.push([states, actions, next_states, rewards, q_values])
            start_state = expert_env.reset()
            state = start_state
            state = Variable(torch.Tensor(state).unsqueeze(0))
            total_reward = 0
            samples = 0
            done = False
            states = []
            next_states = []
            actions = []
            rewards = []
            q_values = []

    def normalize(self):
        for i in range(len(self.memory.memory)):
            batch_states, _, _, _, _ = self.memory.sample_one_at_a_time()
            batch_states = Variable(torch.Tensor(batch_states))
            self.shared_obs_stats.observes(batch_states)

    def update_critic(self, batch_size, num_epoch):
        self.gpu_model.train()
        optimizer = optim.Adam(self.gpu_model.parameters(), lr=10*self.lr)
        #optimizer = RAdam(self.model.parameters(), lr=self.lr*10)
        for k in range(num_epoch):
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_q_values, _ = self.memory.sample(batch_size)
            batch_states = self.shared_obs_stats.normalize(Variable(torch.Tensor(batch_states))).to(device)
            batch_q_values = Variable(torch.Tensor(batch_q_values)).to(device) / self.max_reward.value
            v_pred = self.gpu_model.get_value(batch_states, device=device)
            if self.base_controller is not None:
                v_pred = self.base_controller.get_value(batch_states) + v_pred
            loss_value = (v_pred - batch_q_values)**2
            loss_value = 0.5*torch.mean(loss_value)
            optimizer.zero_grad()
            loss_value.backward(retain_graph=True)
            optimizer.step()
            #print(loss_value)

    def update_actor(self, batch_size, num_epoch, supervised=False):
        model_old = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer, num_contact=self.num_contact).to(device)
        model_old.load_state_dict(self.gpu_model.state_dict())
        model_old.set_noise(self.model.noise)
        self.gpu_model.train()
        optimizer = optim.Adam(self.gpu_model.parameters(), lr=self.lr)
        #optimizer = RAdam(self.model.parameters(), lr=self.lr)

        for k in range(num_epoch):
            batch_states, batch_actions, _, _, batch_q_values, batch_log_probs = self.memory.sample(batch_size)
            #mirror
            batch_mirror_states = np.copy(batch_states)
            
            batch_states = self.shared_obs_stats.normalize(Variable(torch.Tensor(batch_states))).to(device)
            batch_q_values = Variable(torch.Tensor(batch_q_values)).to(device) / self.max_reward.value
            #batch_q_values = self.return_obs_stats.normalize(Variable(torch.Tensor(batch_q_values)))
            batch_actions = Variable(torch.Tensor(batch_actions)).to(device)
            v_pred_old = model_old.get_value(batch_states, device=device)
            if self.base_controller is not None:
                v_pred_old += self.base_controller.get_value(batch_states)
            batch_advantages = (batch_q_values - v_pred_old)
            
            probs = self.gpu_model.calculate_prob_gpu(batch_states, batch_actions)
            probs_old = Variable(torch.Tensor(batch_log_probs)).to(device)#model_old.calculate_prob_gpu(batch_states, batch_actions)
            ratio = (probs - (probs_old)).exp()
            ratio = ratio.unsqueeze(1)
            #print("ratio", ratio)
            #print(probs, probs_old)
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(1-self.params.clip, 1+self.params.clip) * batch_advantages
            loss_clip = -torch.mean(torch.min(surr1, surr2))

            #expert loss
            if supervised is True:
                if k % 1000 == 999:
                    batch_expert_states, batch_expert_actions, _, _, _ = self.expert_trajectory.sample(len(self.expert_trajectory.memory))
                else:
                    batch_expert_states, batch_expert_actions, _, _, _ = self.expert_trajectory.sample(min(batch_size, len(self.expert_trajectory.memory)))
                batch_expert_states = Variable(torch.Tensor(batch_expert_states)).to(device)
                batch_expert_actions = Variable(torch.Tensor(batch_expert_actions)).to(device)
                mu_expert = self.gpu_model.sample_best_actions(batch_expert_states)
                loss_expert = torch.mean((batch_expert_actions-mu_expert)**2)
                print(loss_expert)
            else:
                loss_expert = 0

            #mirror loss
            # (
            #     negation_obs_indices,
            #     right_obs_indices,
            #     left_obs_indices,
            #     negation_action_indices,
            #     right_action_indices,
            #     left_action_indices,
            # ) = self.env.get_mirror_indices()
            
            # batch_mirror_states[:, negation_obs_indices] *= -1
            # rl = np.concatenate((right_obs_indices, left_obs_indices))
            # lr = np.concatenate((left_obs_indices, right_obs_indices))
            # batch_mirror_states[:, rl] = batch_mirror_states[:, lr]

            # #with torch.no_grad():            
            # batch_mirror_actions = self.gpu_model.sample_best_actions(batch_states)
            # if self.base_controller is not None:
            #     batch_mirror_actions = self.base_controller.sample_best_actions(batch_states) + batch_mirror_actions
            # batch_mirror_actions_clone = batch_mirror_actions.clone()
            # batch_mirror_actions_clone[:, negation_action_indices] = batch_mirror_actions[:, negation_action_indices] * -1
            # rl = np.concatenate((right_action_indices, left_action_indices))
            # lr = np.concatenate((left_action_indices, right_action_indices))
            # batch_mirror_actions_clone[:, rl] = batch_mirror_actions[:, lr]
            # batch_mirror_states = Variable(torch.Tensor(batch_mirror_states)).to(device)
            # mirror_mu = self.gpu_model.sample_best_actions(batch_mirror_states)
            # if self.base_controller is not None:
            #     mirror_mu = self.base_controller.sample_best_actions(batch_mirror_states) + mirror_mu
            # mirror_loss = torch.mean((mirror_mu - batch_mirror_actions_clone)**2)

            loss_w = 0#torch.mean(batch_w**2)
            entropy_loss = -self.gpu_model.log_std.mean()
            if supervised:
                total_loss = 1.0*loss_expert
            else:
                total_loss = loss_clip
                #print(total_loss)
            #print("mirror_loss", mirror_loss)
            #print(k, loss_w)
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            #print(torch.nn.utils.clip_grad_norm(self.model.parameters(),1))
            optimizer.step()
        #print(self.shared_obs_stats.mean.data)
        if self.lr > 1e-5:
            self.lr *= 0.99
        else:
            self.lr = 1e-5
        if self.weight > 10:
            self.weight *= 0.99
        if self.weight < 10:
            self.weight = 10.0

    def validation(self):
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_q_values = self.validation_trajectory.sample(300)
        model_old = ActorCriticNet(self.num_inputs, self.num_outputs, self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        batch_states = Variable(torch.Tensor(batch_states))
        batch_q_values = Variable(torch.Tensor(batch_q_values))
        batch_actions = Variable(torch.Tensor(batch_actions))
        mu_old, log_std_old, v_pred_old = model_old(batch_states)
        loss = torch.mean((batch_actions-mu_old)**2)
        if loss.data < self.current_best_validation:
            self.current_best_validation = loss.data
        print("validation error", self.current_best_validation)

    def clear_memory(self):
        self.memory.clear()
        self.value_memory.clear()

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def save_shared_obs_stas(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def save_statistics(self, filename):
        statistics = [self.time_passed, self.num_samples, self.test_mean, self.test_std, self.noisy_test_mean, self.noisy_test_std]
        with open(filename, 'wb') as output:
            pickle.dump(statistics, output, pickle.HIGHEST_PROTOCOL)

    def collect_samples_multithread(self):
        #queue = Queue.Queue()
        import time
        self.start = time.time()
        self.lr = 1e-4
        self.weight = 10
        num_threads = 50
        self.num_samples = 0
        self.time_passed = 0
        score_counter = 0
        total_thread = 0
        max_samples = 25000
        seeds = [
            i * 100 for i in range(num_threads)
        ]
        self.explore_noise = mp.Value("f", -1.5)
        #self.base_noise = np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,  2])
        self.base_noise = np.ones(self.num_outputs)
        noise = self.base_noise * self.explore_noise.value
        #noise[[0, 1, 5, 6]] = -3
        ts = [
            mp.Process(target=self.collect_samples,args=(500,), kwargs={'noise':noise, 'random_seed':seed})
            for seed in seeds
        ]
        for t in ts:
            t.start()
            #print("started")
        self.model.set_noise(noise)
        self.gpu_model.set_noise(noise)
        while score_counter < 100:
            if len(self.noisy_test_mean) % 100 == 1:
                self.save_statistics("stats/walker2d_contact_seed16_Iter%d.stat"%(len(self.noisy_test_mean)))
            #print(self.traffic_light.val.value)
            #if len(self.test_mean) % 100 == 1 and self.test_mean[len(self.test_mean)-1] > 300:
            #   self.save_model("torch_model/multiskill/v4_cassie3dMirrorIter%d.pt"%(len(self.test_mean),))
            # while len(self.memory.memory) < 50000:
            #     if self.counter.get() == num_threads:
            #         for i in range(num_threads):
            #             self.memory.push(self.queue.get())
            #         self.counter.increment()
            #     if len(self.memory.memory) < 50000 and self.counter.get() == num_threads + 1:
            #         self.counter.reset()
            #         self.traffic_light.switch()
            self.save_model(self.model_name)
            while len(self.memory.memory) < max_samples:
                #print(self.counter.get())
                if self.counter.get() == num_threads:
                    for i in range(num_threads):
                        #if random.randint(0, 1) == 0:
                        self.memory.push(self.queue.get())
                        #self.value_memory.push(self.value_queue.get())
                    total_thread += num_threads
                        # else:
                        #     self.memory.push_half(self.queue.get())
                    self.counter.increment()
                if self.counter.get() == num_threads + 1 and len(self.memory.memory) < max_samples:
                    self.traffic_light.switch()
                    self.counter.reset()
            self.num_samples += len(self.memory.memory)
            #while not self.best_score_queue.empty():
            #    self.best_trajectory.push_half(self.best_score_queue.get())
            #self.normalize()
            #self.model.to(device)
            self.gpu_model.load_state_dict(self.model.state_dict())
            self.gpu_model.to(device)
            self.gpu_model.set_noise(self.model.noise)
            if self.base_controller is not None:
                self.base_controller.to(device)
            self.update_critic(min(128, len(self.memory.memory)), (len(self.memory.memory)//3000 + 1)*64)
            self.update_actor(min(128, len(self.memory.memory)), (len(self.memory.memory)//3000 + 1)*64, supervised=False)
            #self.update_critic(128, 2560)
            #self.update_actor(128, 2560, supervised=False)
            self.gpu_model.to("cpu")
            if self.base_controller is not None:
                self.base_controller.to("cpu")
            self.model.load_state_dict(self.gpu_model.state_dict())
            self.clear_memory()
            self.run_test(num_test=2)
            self.run_test_with_noise(num_test=2)
            print(self.num_samples, self.noisy_test_mean[-1])
            if self.noisy_test_mean[-1] > 3500:
                score_counter += 1
            else:
                score_counter = 0
            if self.explore_noise.value > -1.5:
                print("main", self.model.noise)
                self.explore_noise.value *= 1.001
                self.model.noise = self.base_noise * self.explore_noise.value
            print(self.max_reward.value)
            self.plot_statistics()

            self.time_passed = time.time() - self.start
            total_thread = 0
            #print("main", self.model.p_fcs[1].bias.data[0])
            self.traffic_light.switch()
            self.counter.reset()

    def add_env(self, env):
        self.env_list.append(env)

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == '__main__':
    seed = 16
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    import gym
    env = gym.make("Walker2d-v2")
    env.set_contact(1)
    ppo = RL(env, [256, 256], contact=True)
    #ppo.base_controller = ActorCriticNet(ppo.num_inputs, ppo.num_outputs, hidden_layer=[256, 256, 256, 256, 256], num_contact=2)
    #ppo.base_controller.load_state_dict(torch.load("torch_model/StepperOct06.pt"))
    ppo.model_name = "torch_model/walker2d_contact_seed16.pt"
    #ppo.model.load_state_dict(torch.load("torch_model/Stepper256X5_65_10_seed8.pt"))
    #ppo.env.set_difficulty([0.65, 0.65, 20, 20])
    #ppo.max_reward.value = 50

    #with open('torch_model/cassie3dMirror2kHz_shared_obs_stats.pkl', 'rb') as input:
    #    shared_obs_stats = pickle.load(input)
    #ppo.normalize_data()
    #ppo.save_shared_obs_stas("torch_model/cassie_terrain_obs_stats.pkl")
    # ppo.collect_expert_samples(500, "torch_model/Stepper256X5_65_00_seed8.pt", noise=-2.0, difficulty = [0.65, 0])
    # ppo.collect_expert_samples(500, "torch_model/Stepper256X5_75_00_seed8.pt", noise=-2.0, difficulty = [0.75, 0])
    # ppo.collect_expert_samples(500, "torch_model/Stepper256X5_85_00_seed8.pt", noise=-2.0, difficulty = [0.85, 0])
    # ppo.collect_expert_samples(500, "torch_model/Stepper256X5_65_10_seed8.pt", noise=-2.0, difficulty = [0.65, 10])
    #ppo.save_model(ppo.model_name)

    ppo.collect_samples_multithread()

    #ppo.start = t.time()