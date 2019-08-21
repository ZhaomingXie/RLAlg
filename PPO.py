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

from model import ActorCriticNet, Shared_obs_stats, ActorCriticNetMixtureExpert

import statistics
import matplotlib.pyplot as plt
from operator import add, sub

import pickle
import threading
import torch.multiprocessing as mp
import queue
from utils import TrafficLight
from utils import Counter



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
    def __init__(self, env, hidden_layer=[64, 64]):
        self.env = env
        #self.env.env.disableViewer = False
        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        self.hidden_layer = hidden_layer

        self.params = Params()

        self.model = ActorCriticNetMixtureExpert(self.num_inputs, self.num_outputs,self.hidden_layer)
        self.model.share_memory()
        self.shared_obs_stats = Shared_obs_stats(self.num_inputs)
        self.best_model = ActorCriticNet(self.num_inputs, self.num_outputs,self.hidden_layer)
        self.memory = ReplayMemory(self.params.num_steps * 10000)
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

        self.mpdone = [mp.Event(), mp.Event(), mp.Event(), mp.Event()]

        self.process = []
        self.traffic_light = TrafficLight()
        self.counter = Counter()

        self.best_trajectory = ReplayMemory(5000)
        self.best_score_queue = mp.Queue()
        self.best_score = mp.Value("f", 0)
        self.max_reward = mp.Value("f", 1)

        self.expert_trajectory = ReplayMemory(600000)

        self.validation_trajectory = ReplayMemory(6000*9)

        self.best_validation = 1.0
        self.current_best_validation = 1.0

        self.return_obs_stats = Shared_obs_stats(1)

    def normalize_data(self, num_iter=50000, file='shared_obs_stats.pkl'):
        state = self.env.reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        #model_old = ActorCriticNet(self.num_inputs, self.num_outputs,self.hidden_layer)
        #model_old.load_state_dict(self.model.state_dict())
        for i in range(num_iter):
            self.shared_obs_stats.observes(state)
            state = self.shared_obs_stats.normalize(state)
            mu = self.model.sample_actions(state)
            action = mu#(mu + log_std.exp()*Variable(eps))
            env_action = action.data.squeeze().numpy()
            env.action = np.random.randn(self.num_outputs)
            state, reward, done, _ = self.env.step(env_action)

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
                action = mu.data.squeeze().numpy()
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
                action = (mu + 0.1*Variable(eps))
                action = action.data.squeeze().numpy()
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
        #env.seed(random_seed+3)

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
        self.model.set_noise(noise)

        state = start_state
        state = Variable(torch.Tensor(state).unsqueeze(0))
        total_reward = 0
        #q_value = Variable(torch.zeros(1, 1))
        while True:
            signal_init = self.traffic_light.get()
            score = 0
            while samples < num_samples and not done:
                #self.shared_obs_stats.observes(state)

                states.append(state.data.numpy())
                #self.shared_obs_stats.observes(state)
                state = self.shared_obs_stats.normalize(state)
                action = self.model.sample_actions(state)

                actions.append(action.data.numpy())
               
                env_action = action.data.squeeze().numpy()
                state, reward, done, _ = self.env.step(env_action)
                score += reward
                if reward > self.max_reward.value:
                    self.max_reward.value = reward
                #reward *= 0.3
                rewards.append(Variable(reward * torch.ones(1)).data.numpy())
                
                real_rewards.append(Variable(reward * torch.ones(1)).data.numpy())
                
                state = Variable(torch.Tensor(state).unsqueeze(0))
                next_states.append(state.data.numpy())
                next_state = self.shared_obs_stats.normalize(state)
                
                samples += 1
  

            state = self.shared_obs_stats.normalize(state)

            v = (self.model.get_value(state))*self.max_reward.value# / self.return_obs_stats.std) + self.return_obs_stats.mean
            if done:
                R = torch.zeros(1, 1) * -1000
            else:
                R = v.data
            R = Variable(R)
            for i in reversed(range(len(real_rewards))):
                reward = Variable(torch.from_numpy(real_rewards[i]).unsqueeze(0))
                R = self.params.gamma * R + reward#self.return_obs_stats.normalize(reward)# Variable(torch.from_numpy(real_rewards[i]))
                q_values.insert(0, R.data.numpy())
                self.return_obs_stats.observes(R)


            self.queue.put([states, actions, next_states, rewards, q_values])
            self.counter.increment()
            self.env.reset()
            #print(score)
            if score > self.best_score.value:
                self.best_score_queue.put([states, actions, next_states, rewards, q_values])
                self.best_score.value = score
                print("best score", self.best_score.value)
                #self.max_reward.value = self.best_score.value / samples * 99
           
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
            if self.model.noise[0] > -2:
                self.model.noise *= 1.001

    def normalize(self):
        for i in range(len(self.memory.memory)):
            batch_states, _, _, _, _ = self.memory.sample_one_at_a_time()
            batch_states = Variable(torch.Tensor(batch_states))
            self.shared_obs_stats.observes(batch_states)

    def update_critic(self, batch_size, num_epoch):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr*10)
        for k in range(num_epoch):
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_q_values = self.memory.sample(batch_size)
            batch_states = self.shared_obs_stats.normalize(Variable(torch.Tensor(batch_states)))
            batch_q_values = Variable(torch.Tensor(batch_q_values)) / self.max_reward.value
            v_pred = self.model.get_value(batch_states)
            loss_value = (v_pred - batch_q_values)**2
            loss_value = 0.5*torch.mean(loss_value)
            optimizer.zero_grad()
            loss_value.backward(retain_graph=True)
            optimizer.step()
            #print(loss_value)

    def update_actor(self, batch_size, num_epoch, supervised=False):
        model_old = ActorCriticNetMixtureExpert(self.num_inputs, self.num_outputs, self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        model_old.set_noise(self.model.noise)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for k in range(num_epoch):
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_q_values = self.memory.sample(batch_size)

            batch_states = self.shared_obs_stats.normalize(Variable(torch.Tensor(batch_states)))
            batch_q_values = Variable(torch.Tensor(batch_q_values)) / self.max_reward.value
            #batch_q_values = self.return_obs_stats.normalize(Variable(torch.Tensor(batch_q_values)))
            batch_actions = Variable(torch.Tensor(batch_actions))
            v_pred_old = model_old.get_value(batch_states)
            batch_advantages = (batch_q_values - v_pred_old)
            probs_old = model_old.calculate_prob(batch_states, batch_actions)
            probs = self.model.calculate_prob(batch_states, batch_actions)

            mu_old = model_old.get_mean_actions(batch_states)[0]
            mu = self.model.get_mean_actions(batch_states)[0]
            # log_std_old = model_old.get_log_stds(mu_old)
            # log_std = self.model.get_log_stds(mu)
            # probs_old = normal(batch_actions, mu_old, log_std_old)
            # probs = normal(batch_actions, mu, log_std)
            # ratio = (probs.exp()/probs_old.exp())
            # print("ratio1", ratio.mean())
            
            probs = self.model.calculate_prob(batch_states, batch_actions)
            probs_old = model_old.calculate_prob(batch_states, batch_actions)
            ratio = (probs - (probs_old)).exp()
            ratio = ratio.unsqueeze(1)
            # print("ratio", ratio.mean())
            #print(probs, probs_old)
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(1-self.params.clip, 1+self.params.clip) * batch_advantages
            loss_clip = -torch.mean(torch.min(surr1, surr2))

            #expert loss
            if supervised is True:
                if k % 1000 == 999:
                    batch_expert_states, batch_expert_actions, _, _, _ = self.expert_trajectory.sample(len(self.expert_trajectory.memory))
                else:
                    batch_expert_states, batch_expert_actions, _, _, _ = self.best_trajectory.sample(min(batch_size, len(self.best_trajectory.memory)))
                batch_expert_states = Variable(torch.Tensor(batch_expert_states))
                batch_expert_actions = Variable(torch.Tensor(batch_expert_actions))
                mu_expert = self.model.sample_actions(batch_expert_states)
                loss_expert1 = torch.mean((batch_expert_actions-mu_expert)**2)
                loss_expert = loss_expert1#torch.min(loss_expert1, loss_expert2)
            else:
                loss_expert = 0

            batch_w = self.model.get_w(batch_states)
            loss_w = 0#torch.mean(batch_w**2)
            total_loss = 1.0*loss_clip + self.weight*loss_expert + 0.05*loss_w
            #print(k, loss_w)
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            #print(torch.nn.utils.clip_grad_norm(self.model.parameters(),1))
            optimizer.step()
        if self.lr > 1e-4:
            self.lr *= 0.99
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

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def save_shared_obs_stas(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def save_statistics(self, filename):
        statistics = [self.num_samples, self.test_mean, sefl.test_std, self.noisy_test_mean, self.noisy_test_std]
        with open(filename, 'wb') as output:
            pickle.dump(statistics, output, pickle.HIGHEST_PROTOCOL)

    def collect_samples_multithread(self):
        #queue = Queue.Queue()
        self.lr = 1e-4
        self.weight = 10
        num_threads = 50
        self.num_samples = 0
        seeds = [
            np.random.randint(0, 4294967296) for _ in range(num_threads)
        ]
        #noise = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        noise = np.ones(17) * -1
        ts = [
            mp.Process(target=self.collect_samples,args=(500,), kwargs={'noise':noise, 'random_seed':seed})
            for seed in seeds
        ]
        for t in ts:
            t.start()
            #print("started")
        self.model.set_noise(noise)
        while True:
            if len(self.noisy_test_mean) % 100 == 1:
                self.save_statistics("stats/Humanoid_ppo_seed1_Iter%d.stat"%(len(self.noisy_test_mean)))
            self.save_model("torch_model/Humanoid_ppo_seed1.pt")
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

            while len(self.memory.memory) < 50000:
                if self.counter.get() == num_threads:
                    for i in range(num_threads):
                        #if random.randint(0, 1) == 0:
                        self.memory.push(self.queue.get())
                        # else:
                        #     self.memory.push_half(self.queue.get())
                    self.counter.increment()
                if self.counter.get() == num_threads + 1:
                    break
            self.num_samples += len(self.memory.memory)
            while not self.best_score_queue.empty():
                self.best_trajectory.push_half(self.best_score_queue.get())
            #self.normalize()
            self.update_critic(min(128, len(self.memory.memory)), (len(self.memory.memory)//3000 + 1) * 64 * 2)
            #self.memory.memory = list.copy(self.best_trajectory.memory) + self.memory.memory
            self.update_actor(min(128, len(self.memory.memory)), (len(self.memory.memory)//3000 + 1) * 64 * 2, supervised=False)
            self.clear_memory()
            self.run_test(num_test=2)
            self.run_test_with_noise(num_test=2)
            print(self.num_samples, self.test_mean[-1])
            if self.model.noise[0] > -2:
                print(self.model.noise)
                self.model.noise *= 1.001
            self.plot_statistics()
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
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    torch.set_num_threads(1)
    import gym
    env = gym.make("Humanoid-v2")
    ppo = RL(env, [256, 256])

    ppo.collect_samples_multithread()

    start = t.time()