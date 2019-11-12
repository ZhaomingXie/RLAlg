import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.distributions import Normal, Categorical
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActorCriticNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(ActorCriticNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        self.hidden_layer_v = [256, 256, 256]
        p_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        v_fc = nn.Linear(num_inputs, self.hidden_layer_v[0])
        self.p_fcs.append(p_fc)
        self.v_fcs.append(v_fc)
        for i in range(len(self.hidden_layer)-1):
            p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            v_fc = nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1])
            self.p_fcs.append(p_fc)
            self.v_fcs.append(v_fc)
        self.mu = nn.Linear(self.hidden_layer[-1], num_outputs)
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.v = nn.Linear(self.hidden_layer_v[-1],1)
        self.noise = 0
        #self.train()

    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = F.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

    def set_noise(self, noise):
        self.noise = noise

    def get_action(self, inputs):
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = F.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)
        return mu, log_std

    def get_value(self, inputs):
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        return v

class ActorCriticNetMixtureExpert(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], v_hidden_layer=[256, 256], w_hidden_layer=[256, 256], num_expert=1):
        super(ActorCriticNetMixtureExpert, self).__init__()
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.num_expert = num_expert
        self.hidden_layer = hidden_layer
        self.v_hidden_layer = v_hidden_layer
        self.w_hidden_layer = w_hidden_layer
        self.experts = nn.ModuleList()
        for i in range(num_expert):
            expert_i = ActorCriticNet(num_inputs, num_outputs, hidden_layer)
            self.experts.append(expert_i)

        self.v_fcs = nn.ModuleList()
        v_fc = nn.Linear(num_inputs, self.v_hidden_layer[0])
        self.v_fcs.append(v_fc)
        for i in range(len(self.v_hidden_layer)-1):
            v_fc = nn.Linear(self.v_hidden_layer[i], self.v_hidden_layer[i+1])
            self.v_fcs.append(v_fc)
        self.v = nn.Linear(self.v_hidden_layer[-1],1)

        self.w_fcs = nn.ModuleList()
        w_fc = nn.Linear(num_inputs, self.w_hidden_layer[0])
        self.w_fcs.append(w_fc)
        for i in range(len(self.w_hidden_layer)-1):
            w_fc = nn.Linear(self.w_hidden_layer[i], self.w_hidden_layer[i+1])
            self.w_fcs.append(w_fc)
        self.w = nn.Linear(self.w_hidden_layer[-1], num_expert)
        self.noise = -1
        self.log_std = nn.Parameter(torch.ones(num_outputs)*-1,requires_grad=True)

        # for module in self.experts:
        #     from torch.nn import init
        #     for p in module.p_fcs:
        #         #init.xavier_uniform_(p.weight)
        #         init.normal_(p.weight, std=0.2)
                #p.weight.data.uniform_(-0.2, 0.2)

    def get_value(self, inputs):
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.v_hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        return v

    def forward(self, inputs):
        actions = self.get_mean_actions(inputs)
        
        # critic
        v = self.get_value(inputs)

        # w
        w = self.get_w(inputs)
        return actions, v, w

    def get_mean_actions(self, inputs):
        actions = []
        for i in range(self.num_expert):
            a = self.experts[i].get_action(inputs)[0]
            actions.append(a)
        #print(actions[0].shape)
        return actions

    def get_w(self, inputs):
        x = F.relu(self.w_fcs[0](inputs))
        for i in range(len(self.w_hidden_layer)-1):
            x = F.relu(self.w_fcs[i+1](x))
        w = self.w(x)
        #w = torch.ones(inputs.size()[0], self.num_expert)
        w = F.softmax(w, dim=-1)
        #print(w)
        return w

    def sample_actions(self, inputs):
        actions = self.get_mean_actions(inputs)
        w = self.get_w(inputs)
        #print(w)
        categorical = Categorical(w)
        pis = list(categorical.sample().data)
        log_std = self.get_log_stds(actions[0])#Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(actions[0])
        #print(log_std)
        std = torch.exp(log_std)
        sample = Variable(std.data.new(std.size(0), std.size(1)).normal_())
        for i, j in enumerate(pis):
            #print(std[i])
            sample[i] = sample[i].mul(std[i]).add(actions[j][i])
        #sample[:, 0:3] *= 0
        #sample[:, 11:17] *= 0    
        return sample

    def sample_best_actions(self, inputs):
        actions = self.get_mean_actions(inputs)
        w = self.get_w(inputs)
        log_std = self.get_log_stds(actions[0])#log_stds = self.noise * torch.ones(self.num_outputs).expand_as(actions[0])
        #print(log_std)
        eps = torch.randn(actions[0].size())
        values, indices = torch.max(w, 1)
        #print(values ,indices)
        return actions[indices]# + eps * log_stds.exp()

    def calculate_prob(self, inputs, actions):
        log_stds = self.get_log_stds(actions)
        #print(log_stds.shape)
        mean_actions = torch.stack(self.get_mean_actions(inputs))
        #print(mean_actions.shape)
        w = self.get_w(inputs)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        #probs = log_probs.sum(dim=0)
        return probs

    def calculate_prob_gpu(self, inputs, actions):
        log_stds = self.get_log_stds(actions).to(device)
        #print(log_stds.shape)
        mean_actions = torch.stack(self.get_mean_actions(inputs))
        #print(mean_actions.shape)
        w = self.get_w(inputs)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        return probs

    def get_log_stds(self, actions):
        return Variable(torch.Tensor(self.noise)*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(actions)

        #return self.log_std.unsqueeze(0).expand_as(actions)

    def get_actions_difference(self, inputs):
        actions = self.get_mean_actions(inputs)
        difference_matrix = torch.zeros(self.num_expert, self.num_expert)
        for i in range(self.num_expert):
            for j in range(self.num_expert):
                difference_matrix[i, j] =  torch.max((actions[i]-actions[j])**2)
        return difference_matrix

    def set_noise(self, noise):
        self.noise = noise

class ActorCriticNetWithPhase(ActorCriticNet):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], cat_index=46):
        super().__init__(num_inputs, num_outputs, hidden_layer)
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        self.hidden_layer_v = [256, 256, 256]
        p_fc = nn.Linear(46, self.hidden_layer[0])
        v_fc = nn.Linear(num_inputs, self.hidden_layer_v[0])
        self.p_fcs.append(p_fc)
        self.v_fcs.append(v_fc)
        for i in range(len(self.hidden_layer)-1):
            if i == 0:
                p_fc = nn.Linear(self.hidden_layer[i] + 2, self.hidden_layer[i+1])
                v_fc = nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1])
            else:
                p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
                v_fc = nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1])
            self.p_fcs.append(p_fc)
            self.v_fcs.append(v_fc)
        self.mu = nn.Linear(self.hidden_layer[-1], num_outputs)
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.v = nn.Linear(self.hidden_layer_v[-1],1)
        self.cat_index = cat_index
    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs[:, 0:46]))
        for i in range(len(self.hidden_layer)-1):
            if i == 0:
                x = torch.cat([x, inputs[:, self.cat_index:self.cat_index+2]], 1)
            x = F.relu(self.p_fcs[i+1](x))
        mu = F.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

class ActorCriticNetWithBias(ActorCriticNet):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], cat_index=46):
        super().__init__(num_inputs, num_outputs, hidden_layer)
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        self.hidden_layer_v = [256, 256]
        p_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        v_fc = nn.Linear(num_inputs, self.hidden_layer_v[0])
        self.p_fcs.append(p_fc)
        self.v_fcs.append(v_fc)
        for i in range(len(self.hidden_layer)-1):
            if i == 0:
                p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
                v_fc = nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1])
            else:
                p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
                v_fc = nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1])
            self.p_fcs.append(p_fc)
            self.v_fcs.append(v_fc)
        self.mu = nn.Linear(self.hidden_layer[-1], num_outputs)
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.v = nn.Linear(self.hidden_layer_v[-1],1)
        self.cat_index = cat_index
    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = F.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

class ActorCriticNetWithHeightMap(ActorCriticNet):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(ActorCriticNetWithHeightMap, self).__init__()
        self.height_map_dim = [512, 512]
        self.height_map = np.zeros((512, 512))

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 11, stride=3, padding=0),nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 9, stride=3, padding=0),nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=3, padding=1),nn.ReLU())
        self.fc = nn.Linear(5*5*32, 128)
        self.p_fcs[0] = nn.Linear(num_inputs + 128, self.hidden_layer[0])
        self.v_fcs[0] = nn.Linear(num_inputs + 128, self.hidden_layer[0])

    def set_height_map(self, height_map):
        self.height_map = np.copy(height_map)

    def forward(self, input):
        hegiht_map = Variable(torch.Tensor(self.hegiht_map).unsqueeze(0))
        height_map_output = self.conv1(height_map)
        height_map_output = self.conv2(height_map_output)
        height_map_output = self.conv3(height_map_output)
        height_map_output = self.out_view(out.size(0), -1)
        height_map_output = self.fc(height_map_output)
        x = F.relu(self.p_fc[0](torch.cat([input, height_map_output])))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = F.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](torch.cat([inputs2, height_map_output])))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

class ActorCriticNetWithMultiRobot(ActorCriticNet):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], num_robots=1):
        #super().__init__(num_inputs, num_outputs, hidden_layer=hidden_layer)
        super(ActorCriticNet, self).__init__()
        self.hidden_layer_v = [256, 256]
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.noise = 0
        
        #robot specific modules
        self.robot_policy_input_modules = nn.ModuleList()
        self.robot_value_input_modules = nn.ModuleList()
        self.robot_policy_output_modules = nn.ModuleList()
        self.robot_value_output_modules = nn.ModuleList()
        for i in range(num_robots):
            p_fc = nn.Linear(num_inputs, self.hidden_layer[0])
            v_fc = nn.Linear(num_inputs, self.hidden_layer_v[0])
            p_fc.weight.data.fill_(0.01)
            p_fc.bias.data.fill_(0.01)
            v_fc.weight.data.fill_(0.01)
            v_fc.bias.data.fill_(0.01)
            self.robot_policy_input_modules.append(p_fc)
            self.robot_value_input_modules.append(v_fc)

            p_fc = nn.Linear(self.hidden_layer[-1], self.num_outputs)
            v_fc = nn.Linear(self.hidden_layer_v[-1], 1)
            p_fc.weight.data.fill_(0.01)
            p_fc.bias.data.fill_(0.01)
            v_fc.weight.data.fill_(0.01)
            v_fc.bias.data.fill_(0.01)
            self.robot_policy_output_modules.append(p_fc)
            self.robot_value_output_modules.append(v_fc)

        #common modules
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        for i in range(len(self.hidden_layer)-1):
            p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            self.p_fcs.append(p_fc)
        for i in range(len(self.hidden_layer_v)-1):
            v_fc = nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1])
            self.v_fcs.append(v_fc)

    def compute_robot_action(self, inputs, robot_id=0):
        x = F.relu(self.robot_policy_input_modules[robot_id](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i](x))
        mu = F.tanh(self.robot_policy_output_modules[robot_id](x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        x = F.relu(self.robot_value_input_modules[robot_id](inputs))
        for i in range(len(self.hidden_layer_v)-1):
            x = F.relu(self.v_fcs[i](x))
        v = (self.robot_value_output_modules[robot_id](x))
        return mu, log_std, v

    #by default, use the first robot's parameter to do inference
    def forward(self, inputs):
        return self.compute_robot_action(inputs, robot_id=0)

    def one_step_inference(self, inputs, robot_id=0):
        x_policy = F.relu(self.robot_policy_input_modules[robot_id](inputs))
        x_value = F.relu(self.robot_value_input_modules[robot_id](inputs))
        return x_policy, x_value

class ActorNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(ActorNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.log_stds = nn.ModuleList()
        p_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        log_std = nn.Linear(num_inputs, self.hidden_layer[0])
        self.p_fcs.append(p_fc)
        self.log_stds.append(log_std)
        for i in range(len(self.hidden_layer)-1):
            p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            log_std = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            self.p_fcs.append(p_fc)
            self.log_stds.append(log_std)
        self.mu = nn.Linear(self.hidden_layer[-1], num_outputs)
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.noise = -2.0
        self.noises = torch.Tensor(num_outputs)
        self.log_std_linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        #log_std = F.relu(self.log_stds[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
            #log_std = F.relu(self.log_stds[i+1](log_std))
        #print(self.mu(x))
        mu = F.softsign(self.mu(x))
        #print(mu)
        log_std = Variable(self.noise * torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)
        #log_std.to(device)
        #log_std = F.tanh((self.log_std_linear(inputs)))
        #log_std = torch.clamp(log_std, min=-2, max=2)
        return mu, log_std
    def sample_gpu(self, inputs):
        mean, log_std = self.forward(inputs)
        #mean.to(device)
        std = log_std.exp().to(device)
        eps = torch.randn(mean.shape).to(device)
        normal = Normal(mean, std)
        #action = normal.rsample()
        action = (mean + (std * eps).clamp(-0.5, 0.5)).clamp(-1.0, 1.0)#torch.clamp(normal.rsample(), -1.0, 1.0)
        return (action), 0, (mean), 0
    def sample(self, inputs):
        mean, log_std = self.forward(inputs)
        #mean.to(device)
        std = log_std.exp()
        eps = torch.randn(mean.shape)
        normal = Normal(mean, std)
        #action = normal.rsample()
        action = (mean + (std * eps).clamp(-0.5, 0.5)).clamp(-1.0, 1.0)#torch.clamp(normal.rsample(), -1.0, 1.0)
        return (action), 0, (mean), 0
    def set_noise(self, noise):
        self.noise = noise

class ValueNet(nn.Module):
    def __init__(self, num_inputs, hidden_layer=[64, 64]):
        super(ValueNet, self).__init__()
        self.hidden_layer = hidden_layer
        self.v_fcs = nn.ModuleList()
        v_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        self.v_fcs.append(v_fc)
        for i in range(len(self.hidden_layer)-1):
            v_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            self.v_fcs.append(v_fc)
        self.v = nn.Linear(self.hidden_layer[-1],1)
    def forward(self, inputs):
        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return v

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(QNet, self).__init__()
        self.hidden_layer = hidden_layer
        self.num_outputs = num_outputs
        self.q_fcs1 = nn.ModuleList()
        self.q_fcs2 = nn.ModuleList()
        q_fc1 = nn.Linear(num_inputs + num_outputs, self.hidden_layer[0])
        q_fc2 = nn.Linear(num_inputs + num_outputs, self.hidden_layer[0])
        self.q_fcs1.append(q_fc1)
        self.q_fcs2.append(q_fc2)
        for i in range(len(self.hidden_layer)-1):
            q_fc1 = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            q_fc2 = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            self.q_fcs1.append(q_fc1)
            self.q_fcs2.append(q_fc2)
        self.q_1 = nn.Linear(self.hidden_layer[-1],1)
        self.q_2 = nn.Linear(self.hidden_layer[-1],1)
    def forward(self, states, actions):
        inputs = torch.cat([states, actions], 1)
        q1 = F.relu(self.q_fcs1[0](inputs))
        q2 = F.relu(self.q_fcs2[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            q1 = F.relu(self.q_fcs1[i+1](q1))
            q2 = F.relu(self.q_fcs2[i+1](q2))
        q1 = (self.q_1(q1))
        q2 = (self.q_2(q2))
        return q1, q2


class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] += p.grad.data

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)

class Shared_obs_stats():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.mean_diff = torch.zeros(num_inputs).share_memory_()
        self.std = torch.ones(num_inputs).share_memory_()
        self.num_inputs = num_inputs
        self.sum = torch.zeros(num_inputs).share_memory_()
        self.sum_sqr = torch.zeros(num_inputs).share_memory_()

    def observes(self, obs):
        # observation mean var updates
        x = obs.data.squeeze()
        if True:
            self.n += 1.
            last_mean = self.mean.clone()
            self.sum = self.sum + x
            self.sum_sqr += x.pow(2)
            self.mean = self.sum / self.n
            self.std = (self.sum_sqr / self.n - self.mean.pow(2)).clamp(1e-2,1e9).sqrt()
            self.mean = self.mean.float()
            self.std = self.std.float()
        #self.mean = (self.mean * self.n + x) / self.
            #self.mean += (x-self.mean)/self.n
            #self.mean_diff += (x-last_mean)*(x-self.mean)
            #self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        #if (inputs.shape[1]) > self.num_inputs:
        #    inputs = inputs[:, 0:self.num_inputs]
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs[:, 0:self.num_inputs]))
        obs_std = Variable(self.std.unsqueeze(0).expand_as(inputs[:, 0:self.num_inputs]))
        obs_mean = ((inputs[:, 0:self.num_inputs] - obs_mean) / obs_std)
        if (inputs.shape[1]) > self.num_inputs:
            #print(inputs.shape, obs_mean.shape)
            obs_mean = torch.cat([obs_mean, inputs[:, self.num_inputs:self.num_inputs+1]], dim=1)
        #print("outout", obs_mean)
        #obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp(obs_mean, -10.0, 10.0)

    def reset(self):
        self.n = torch.zeros(self.num_inputs).share_memory_()
        self.mean = torch.zeros(self.num_inputs).share_memory_()
        self.mean_diff = torch.zeros(self.num_inputs).share_memory_()
        self.var = torch.zeros(self.num_inputs).share_memory_()