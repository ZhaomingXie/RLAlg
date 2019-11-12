from PPO import *
from TD3 import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", required=True, type=str)
parser.add_argument("--stats_path", required=True, type=str)
parser.add_argument("--env", required=True, type=str)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--learn_contact", action='store_true')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(args.seed)
import gym
env = gym.make(args.env)
env.seed(args.seed)
ppo = RL(env, [256, 256], learn_contact=args.learn_contact)
print(args.learn_contact)
ppo.seed = args.seed
ppo.model_name = args.policy_path
ppo.stats_name = args.stats_path
ppo.save_model(ppo.model_name)
ppo.collect_samples_multithread()