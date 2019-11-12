from PPO import *
from model import *
import pybullet_envs
from radam import RAdam
#from cassieRLEnvMirrorPhase import *

from raisim_gym.env.RaisimGymVecEnv import RaisimGymVecEnv as Environment
from raisim_gym.env.env.ANYmal import __ANYMAL_RESOURCE_DIRECTORY__ as __RSCDIR__
# try:
# 	mp.set_start_method('spawn')
# except RuntimeError:
# 	pass

# from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass
#from multiprocessing import set_start_method
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RL_TD3(RL):
	def __init__(self, env, hidden_layer=[64, 64]):
		super().__init__(env, hidden_layer=hidden_layer)
		self.env = env
		self.num_inputs = env.observation_space.shape[0]
		self.num_outputs = env.action_space.shape[0]
		self.hidden_layer = hidden_layer
		self.params = Params()
		self.actor = ActorNet(self.num_inputs, self.num_outputs,self.hidden_layer).to("cpu")
		self.actor_target = ActorNet(self.num_inputs, self.num_outputs, self.hidden_layer).to("cpu")
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.q_function = QNet(self.num_inputs, self.num_outputs,self.hidden_layer).to("cpu")
		self.q_function_target = QNet(self.num_inputs, self.num_outputs,self.hidden_layer).to("cpu")
		self.q_function_target.load_state_dict(self.q_function.state_dict())
		#self.actor_target = ActorNet(self.num_inputs, num_outputs, self.hidden_layer)
		#self.actor_target.load_state_dict(self.actor.state_dict())
		self.q_function.share_memory()
		self.actor.share_memory()
		self.q_function_target.share_memory()
		self.actor_target.share_memory()
		self.shared_obs_stats = Shared_obs_stats(self.num_inputs)
		self.memory = ReplayMemory(1e8)
		self.test_mean = []
		self.test_std = []
		self.lr = 1e-4
		plt.show(block=False)
		self.test_list = []
		
		#for multiprocessing queue
		self.queue = mp.Queue()
		self.process = []
		self.traffic_light = TrafficLight()
		self.counter = Counter()

		self.off_policy_memory = ReplayMemory(1e8)

		#self.q_function_optimizer = optim.Adam(self.q_function.parameters(), lr=self.lr, weight_decay=0e-3)
		#self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=0e-3)
		self.q_function_optimizer = RAdam(self.q_function.parameters(), lr=self.lr, weight_decay=0e-3)
		self.actor_optimizer = RAdam(self.actor.parameters(), lr=self.lr, weight_decay=0e-3)
		self.actor.train()
		self.q_function.train()

		self.q_fucntion_scheduler = optim.lr_scheduler.ExponentialLR(self.q_function_optimizer, gamma=0.99)
		self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.99)

		self.off_policy_queue = mp.Queue()

		self.reward_scale = 1
		self.max_reward = mp.Value("f", 50)
		#self.actor_target.share_memory()

	def run_test(self, num_test=1):
		state = self.env.reset()#_for_test()
		state = Variable(torch.Tensor(state).unsqueeze(0))
		ave_test_reward = 0

		total_rewards = []
		#actor_old = ActorNet(self.num_inputs, self.num_outputs, self.hidden_layer)
		#actor_old.load_state_dict(self.actor.state_dict())
		
		for i in range(num_test):
			total_reward = 0
			num_step = 0
			while True:
				state = self.shared_obs_stats.normalize(state).to("cpu")
				#print(self.actor.device)
				action, _, mean, log_std = self.actor.sample(state)
				#print("log std", log_std)
				print(mean.data)
				env_action = mean.cpu().data.squeeze().numpy()
				state, reward, done, _ = self.env.step(env_action)

				total_reward += reward
				num_step += 1
				#done = done or (num_step > 400)

				if done:
					state = self.env.reset()#_for_test()
					#print(self.env.position)
					#print(self.env.time)
					state = Variable(torch.Tensor(state).unsqueeze(0))
					ave_test_reward += total_reward / num_test
					total_rewards.append(total_reward)
					break
				state = Variable(torch.Tensor(state).unsqueeze(0))

		reward_mean = statistics.mean(total_rewards)
		reward_std = statistics.stdev(total_rewards)
		self.test_mean.append(reward_mean)
		self.test_std.append(reward_std)
		self.test_list.append((reward_mean, reward_std))

	def plot_statistics(self): 
		ax = self.fig.add_subplot(111)
		low = []
		high = []
		index = []
		for i in range(len(self.test_mean)):
			low.append(self.test_mean[i] - self.test_std[i])
			high.append(self.test_mean[i] + self.test_std[i])
			index.append(i)
		#ax.set_xlim([0,1000])
		#ax.set_ylim([0,300])
		plt.xlabel('iterations')
		plt.ylabel('average rewards')
		ax.plot(self.test_mean, 'b')
		ax.fill_between(index, low, high, color='cyan')
		#ax.plot(map(sub, test_mean, test_std))
		self.fig.canvas.draw()

	def collect_samples(self, num_samples, noise=-2.0, random_seed=1):
		torch.set_num_threads(1)
		#print(random_seed)
		#env.seed(random_seed+3)
		#random seed is used to make sure different thread generate different trajectories
		random.seed(random_seed)
		torch.manual_seed(random_seed+1)
		np.random.seed(random_seed+2)
		#torch.cuda.manual_seed_all(random_seed+3)
		start_state = self.env.reset()
		samples = 0
		done = False
		states = []
		next_states = []
		actions = []
		rewards = []
		q_values = []
		dones = []
		self.actor.set_noise(noise)
		state = start_state
		
		state = Variable(torch.Tensor(state).unsqueeze(0))
		total_reward = 0
		start = t.time()
		while True:
			#actor_old = ActorNet(self.num_inputs, self.num_outputs, self.hidden_layer)
			#actor_old.load_state_dict(self.actor.state_dict())
			#print("something")
			self.actor.load_state_dict(torch.load(self.model_path))
			signal_init = self.traffic_light.get()
			while samples < num_samples and not done:
				state = self.shared_obs_stats.normalize(state).to("cpu")
				states.append(state.cpu().data.numpy())
				if self.traffic_light.explore.value == False:# and random.randint(0,90)%100 > 0:
					action, _, mean, _ = self.actor.sample(state)
					action.detach()
					mean.detach()
					#print(action)
				else:
					action = np.random.randint(-100, 100, size=(self.env.action_space.shape[0],))*1.0/100.0
					#action = self.env.action_space.sample()
					action = Variable(torch.Tensor(action).unsqueeze(0))
				actions.append(action.cpu().data.numpy())
				env_action = action.cpu().data.squeeze().numpy()

				state, reward, done, _ = self.env.step(env_action)
				if reward > self.max_reward.value:
					self.max_reward.value = min(reward, 50.0)
				#print(env_action)
				#print(samples, env_action, reward, state)

				#print(reward)
				total_reward += reward
				#print(samples, total_reward)
				rewards.append(Variable(reward * torch.ones(1, 1)).data.numpy())
				state = Variable(torch.Tensor(state).unsqueeze(0))
				#print(state.shape)
				next_state = self.shared_obs_stats.normalize(state)
				next_states.append(next_state.cpu().data.numpy())
				dones.append(Variable((1 - done) * torch.ones(1, 1)).data.numpy())
				samples += 1
				#done = (done or samples > num_samples)

			self.queue.put([states, actions, next_states, rewards, dones])
			#print(self.actor.p_fcs[0].bias.data[0])
			self.counter.increment()
				#print("waiting sim time passed", t.time() - start)
			start = t.time()
			while self.traffic_light.get() == signal_init:
				pass
			start = t.time()
			state = self.env.reset()
			state = Variable(torch.Tensor(state).unsqueeze(0))
			samples = 0
			print(total_reward)
			total_reward = 0
			done = False
			states = []
			next_states = []
			actions = []
			rewards = []
			values = []
			q_values = []
			dones = []

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
		dones = []
		model_expert = self.Net(self.num_inputs, self.num_outputs,[256, 256, 256, 256, 256])

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
			max_sample = 10000
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
				dones.append(Variable((1 - done) * torch.ones(1, 1)).data.numpy())
				rewards.append(Variable(reward * torch.ones(1, 1)).data.numpy())
				state = Variable(torch.Tensor(state).unsqueeze(0))

				next_state = self.shared_obs_stats.normalize(state)
				next_states.append(next_state.data.numpy())

				samples += 1
				#total_sample += 1
				score += reward
			print("expert score", score)
			# state = self.shared_obs_stats.normalize(state)
			# v = model_expert.get_value(state)
			# if done:
			#     R = torch.zeros(1, 1)
			# else:
			#     R = v.data
			#     R = torch.ones(1, 1) * 100
			# R = Variable(R)
			# for i in reversed(range(len(rewards))):
			#     R = self.params.gamma * R + Variable(torch.from_numpy(rewards[i]))
			#     q_values.insert(0, R.data.numpy())
			
			if not validation and score >= num_samples:
				self.off_policy_memory.push([states, actions, next_states, rewards, dones])
				total_sample += num_samples
			elif score >= num_samples:
				self.validation_trajectory.push([states, actions, next_states, rewards, dones])
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

	def update_q_function(self, batch_size, num_epoch, update_actor=False):
		for k in range(num_epoch):
			batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.off_policy_memory.sample(batch_size)
			# batch_states2, batch_actions2, batch_next_states2, batch_rewards2, batch_dones2 = self.memory.sample(self.num_threads)

			batch_states = Variable(torch.Tensor(batch_states)).to(device)
			batch_next_states = Variable(torch.Tensor(batch_next_states)).to(device)
			batch_actions = Variable(torch.Tensor(batch_actions)).to(device)
			batch_rewards = Variable(torch.Tensor(batch_rewards / self.max_reward.value)).to(device)
			batch_dones = Variable(torch.Tensor(batch_dones)).to(device)

			# batch_states2 = Variable(torch.Tensor(batch_states2))
			# batch_next_states2 = Variable(torch.Tensor(batch_next_states2))
			# batch_actions2 = Variable(torch.Tensor(batch_actions2))
			# batch_rewards2 = Variable(torch.Tensor(batch_rewards2 * self.reward_scale))
			# batch_dones2 = Variable(torch.Tensor(batch_dones2))

			# batch_states = torch.cat([batch_states, batch_states2], 0).to(device)
			# batch_next_states = torch.cat([batch_next_states, batch_next_states2], 0).to(device)
			# batch_actions = torch.cat([batch_actions, batch_actions2], 0).to(device)
			# batch_rewards = torch.cat([batch_rewards, batch_rewards2], 0).to(device)
			# batch_dones = torch.cat([batch_dones, batch_dones2], 0).to(device)

			#compute on policy actions for next state
			batch_next_state_action, batch_next_log_prob,  batch_next_state_action_mean, _, = self.actor_target.sample_gpu(batch_next_states)
			#compute q value for these actions
			q_next_1_target, q_next_2_target = self.q_function_target(batch_next_states, batch_next_state_action)
			q = torch.min(q_next_1_target, q_next_2_target)

			#value functions estimate of the batch_states
			value = batch_rewards + batch_dones * self.params.gamma * q
			
			#q value estimate
			q1, q2 = self.q_function(batch_states, batch_actions)
			#print(q1.shape, value.shape)
			q1_value_loss = F.mse_loss(q1, value)
			q2_value_loss = F.mse_loss(q2, value)
			q_value_loss = q1_value_loss + q2_value_loss
			#print(q_value_loss)

			self.q_function_optimizer.zero_grad()
			q_value_loss.backward()
			self.q_function_optimizer.step()

			if update_actor is False:
				continue

			mean_action, log_std = self.actor(batch_states)
			q1_new, q2_new = self.q_function(batch_states, mean_action)
			new_q_value = torch.min(q1_new, q2_new)# - self.critic(batch_states)
			policy_loss = (-new_q_value).mean() + self.action_weight *(mean_action**2).mean()
			#print("policy_loss",  (-new_q_value).mean())
			#print("log_prob", log_prob.shape, new_q_value.shape)
			
			self.actor_optimizer.zero_grad()
			policy_loss.backward()
			self.actor_optimizer.step()

	def update_q_target(self, tau):
		for target_param, param in zip(self.q_function_target.parameters(), self.q_function.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
	def update_actor_target(self, tau):
		for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
	def save_actor(self, filename):
		torch.save(self.actor.state_dict(), filename)

	def collect_samples_multithread(self):
		import time
		self.num_samples = 0
		self.start = time.time()
		self.num_threads = 1
		self.action_weight = 0.01
		self.lr = 1e-4
		self.traffic_light.explore.value = True
		self.time_passed = 0
		max_samples = 0
		seeds = [
			np.random.randint(0, 4294967296) for _ in range(self.num_threads)
		]

		ts = [
			mp.Process(target=self.collect_samples,args=(500,), kwargs={'noise':-2.0, 'random_seed':seed})
			for seed in seeds
		]
		for t in ts:
			t.start()
			
		for iter in range(1000000):
			while len(self.memory.memory) < max_samples:
				if self.counter.get() == self.num_threads:
					for i in range(self.num_threads):
						#if random.randint(0, 1) == 0:
						self.memory.push(self.queue.get())
						# else:
						#     self.memory.push_half(self.queue.get())
					self.counter.increment()
				if self.counter.get() == self.num_threads + 1:
					break
			print(len(self.memory.memory))
			off_policy_memory_len = len(self.off_policy_memory.memory)
			#print(off_policy_memory_len)
			memory_len = len(self.memory.memory)
			#print(len(self.memory.memory))
			#self.update_critic(128, 640 * int(memory_len/3000))
			if off_policy_memory_len >= 128:
				if off_policy_memory_len > 10000:
					self.traffic_light.explore.value = False
				else:
					print("explore")
				self.actor.to(device)
				self.q_function.to(device)
				self.actor_target.to(device)
				self.q_function_target.to(device)
				#for policy_update in range(len(self.memory.memory)):
				for policy_update in range(32):
					if policy_update % 2 == 0:
						self.update_q_function(128, 100)
					else:
						self.update_q_function(128, 1, update_actor=True)
						self.update_q_target(0.005)
						self.update_actor_target(0.005)
				self.actor.to("cpu")
				self.q_function.to("cpu")
				self.actor_target.to("cpu")
				self.q_function_target.to("cpu")

				#print(self.actor.p_fcs[0].bias.data[0])
				self.num_samples += memory_len
				self.save_actor(self.model_path)
				if iter % 10 == 0:
					self.run_test(num_test=2)
					self.plot_statistics()
					print(self.num_samples, self.test_mean[-1])
				
			self.off_policy_memory.memory = self.off_policy_memory.memory + self.memory.memory
			#if (math.isnan(len(self.memory.memory))):
			#	print(self.memory.memory)
			#print(self.off_policy_memory.memory)
			self.clear_memory()
			self.off_policy_memory.clean_memory()
			#start = t.time()
			#print("waiting memory collectd time passed", t.time() - start)
			self.traffic_light.switch()
			self.counter.reset()

if __name__ == '__main__':
	torch.set_num_threads(1)
	random.seed(1)
	torch.manual_seed(1)
	np.random.seed(1)
	torch.cuda.manual_seed_all(1)
	#from ruamel.yaml import YAML, dump, RoundTripDumper
	#from _raisim_gym import RaisimGymEnv
	#parser = argparse.ArgumentParser()
	#parser.add_argument('--cfg', type=str, default=os.path.abspath(__RSCDIR__ + "/default_cfg.yaml"),
	#                    help='configuration file')
	#cfg_abs_path = parser.parse_args().cfg
	#cfg = YAML().load(open(cfg_abs_path, 'r'))
	import gym
	env = gym.make("mocca_envs:Walker3DStepperEnv-v0")#Environment(RaisimGymEnv(__RSCDIR__, dump(cfg['environment'], Dumper=RoundTripDumper)))
	env.seed(1)
	sac = RL_TD3(env, [256, 256])
	sac.model_path = "torch_model/stepper_TD3_master.pt"
	sac.save_actor(sac.model_path)
	sac.collect_expert_samples(500, "torch_model/Stepper256X5_65_00_seed8.pt", noise=-2.0, difficulty = [0.65, 0])
	sac.collect_expert_samples(500, "torch_model/Stepper256X5_75_00_seed8.pt", noise=-2.0, difficulty = [0.75, 0])
	sac.collect_expert_samples(500, "torch_model/Stepper256X5_85_00_seed8.pt", noise=-2.0, difficulty = [0.85, 0])
	sac.collect_expert_samples(500, "torch_model/Stepper256X5_65_10_seed8.pt", noise=-2.0, difficulty = [0.65, 10])

	sac.env.set_difficulty([0.65, 10])

	sac.collect_samples_multithread()

	start = t.time()

	noise = -2.0