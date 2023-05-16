from collections import defaultdict
import torch
import matplotlib
from torch.optim import Adam
from torch.nn import GRU, Linear
import torch
import matplotlib
from torch.optim import Adam
import gym
from gym import spaces
import time
from copy import deepcopy
import itertools
from core import *
from param import Param
import pickle
import concurrent.futures as futures
from pettingzoo.sisl import job_match_v1
#from ppo import DefaultPolicy
 
def shuffle(comm, k):
    choice = np.random.choice(comm.shape[0], k, replace=False)   
    return comm[choice]
                    
def make_jobmatch_env(**kwargs):
# def make_jobmatch_env(n_freelancers, n_recruiters, 
#                   max_cycles=500):
    env_1 = job_match_v1.parallel_env_1(**kwargs)
    env_2 = job_match_v1.parallel_env_2(**kwargs)
    env_3 = job_match_v1.parallel_env_3(**kwargs)
    env_1.reset()
    env_2.reset()
    env_3.reset()
    
    recruiters = ["recruiter_{}".format(i) for i in range(kwargs["n_agents_recruiters"])]
    freelancers = ["freelancer_{}".format(i) for i in range(kwargs["n_agents_freelancers"])]
    
    return env_1, env_2, env_3, recruiters, freelancers

def main(env_fn=None, actor_critic=MLPActorCritic, ac_kwargs=dict(), trained_dir=None, 
        # beta=False, comm=False, dist_action=False, render=False, 
        recurrent=False, # adv_train=False, ablate_kwargs=None, count_agent=None, 
        max_ep_len=200, epochs=20, # num_agent=9, num_adv=0, 
        # random_attacker=False, test_random = False
        ):

    
    print('-----Test Performance {} Epochs---'.format(epochs))
    
    # Instantiate environment
    env_1, env_2, env_3, recruiters, freelancers = env_fn()
    default_recruiter_name = recruiters[0]
    default_freelancer_name = freelancers[0]
    print("Freelancers:{}".format(recruiters))
    print("Recruiters:{}".format(freelancers))
    
    observation_space_1 = env_1.observation_spaces[default_freelancer_name]
    observation_space_2 = env_2.observation_spaces[default_recruiter_name]
    observation_space_3 = env_3.observation_spaces[default_freelancer_name]
    
    action_space_1 = env_1.action_spaces[default_freelancer_name]
    action_space_2 = env_2.action_spaces[default_recruiter_name]
    action_space_3 = env_3.action_spaces[default_freelancer_name]

    # print("Observation space 1:{}".format(observation_space_1))
    # print("Observation space 2:{}".format(observation_space_2))
    # print("Observation space 3:{}".format(observation_space_3))
    # print("Action space 1:{}".format(action_space_1))
    # print("Action space 2:{}".format(action_space_2))
    # print("Action space 3:{}".format(action_space_3))
    
    act_dim_1 = action_space_1.shape    # multibinary
    act_dim_2 = action_space_2.shape    # box
    act_dim_3 = action_space_3.shape    # discrete

    obs_dim_1 = spaces.utils.flatdim(observation_space_1)
    obs_dim_2 = spaces.utils.flatdim(observation_space_2)
    obs_dim_3 = spaces.utils.flatdim(observation_space_3)

    # print("Obs 1 dim:{}, Act 1 dim:{}".format(obs_dim_1, act_dim_1))
    # print("Obs 2 dim:{}, Act 2 dim:{}".format(obs_dim_2, act_dim_2))
    # print("Obs 3 dim:{}, Act 3 dim:{}".format(obs_dim_3, act_dim_3))
    
    high = torch.from_numpy(action_space_2.high).to(Param.device).type(Param.dtype)
    low = torch.from_numpy(action_space_2.low).to(Param.device).type(Param.dtype)
    #print("Obs dim:{}, Act dim:{}".format(obs_dim[0], act_dim))
    
    # Create actor-critic module
    ac_kwargs['beta'] = False
    ac_1 = actor_critic(obs_dim_1, action_space_1, **ac_kwargs).to(Param.device)
    ac_kwargs['beta'] = True
    ac_2 = actor_critic(obs_dim_2, action_space_2, **ac_kwargs).to(Param.device)
    ac_kwargs['beta'] = False
    ac_3 = actor_critic(obs_dim_3, action_space_3, **ac_kwargs).to(Param.device)
    if trained_dir is not None:
        print("loaded pretrained model from", trained_dir)
        ac1_name = os.path.join(trained_dir, "ac_1.pt")
        ac2_name = os.path.join(trained_dir, "ac_2.pt")
        ac3_name = os.path.join(trained_dir, "ac_3.pt")
        # state_dict, mean, std = torch.load(trained_dir, map_location=Param.device)
        state_dict_1, mean_1, std_1 = torch.load(ac1_name, map_location=Param.device)
        state_dict_2, mean_2, std_2 = torch.load(ac2_name, map_location=Param.device)
        state_dict_3, mean_3, std_3 = torch.load(ac3_name, map_location=Param.device)
        ac_1.load_state_dict(state_dict_1)
        ac_1.moving_mean = mean_1
        ac_1.moving_std = std_1
        
        ac_2.load_state_dict(state_dict_2)
        ac_2.moving_mean = mean_2
        ac_2.moving_std = std_2
        
        ac_3.load_state_dict(state_dict_3)
        ac_3.moving_mean = mean_3
        ac_3.moving_std = std_3
    
    mean_f, std_f, mean_r, std_r = test_return(env_1, env_2, env_3, ac_1, ac_2, ac_3, epochs, max_ep_len, freelancers, recruiters,
                            recurrent)
    print('-----[Number of Agents:{} Number of Adversary:{}] {} Epochs Performance:------'.\
          format(len(recruiters), len(freelancers), epochs))
    print('Freelancer: mean:{:.2f}, std:{:.2f}'.format(mean_f, std_f))
    print('Recruiter: mean:{:.2f}, std:{:.2f}'.format(mean_r, std_r))
   
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action="store_true")
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--trained-dir', type=str)
    
    ### environment setting
    parser.add_argument('--n-freelancers', type=int, default=3)
    parser.add_argument('--n-recruiters', type=int, default=3)
    parser.add_argument('--max-cycle', type=int, default=200)
    
    parser.add_argument('--recurrent', action="store_true")
    parser.add_argument('--obs-normalize', action="store_true")
    parser.add_argument('--exp-no', type=int, default=0)

    args = parser.parse_args()
    gym.logger.set_level(40)

    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.cuda)))
    
    budget, num_of_skills, pay_low, pay_high, rate_freelancer, rate_recruiter, base_price, u_ij, v_ij = get_parameters(args.n_freelancers, args.n_recruiters, args.exp_no)    
    env = make_jobmatch_env(budget=budget, num_of_skills=num_of_skills, pay_low=pay_low, pay_high=pay_high,
                            rate_freelancer=rate_freelancer, rate_recruiter=rate_recruiter, base_price=base_price, u_ij=u_ij, v_ij=v_ij,
                            max_cycles=args.max_cycle, n_agents_freelancers=args.n_freelancers, n_agents_recruiters=args.n_recruiters)
    
        
    main(lambda:env, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, # beta=args.beta, 
                       recurrent=args.recurrent, ep_len=args.max_cycle), 
        epochs=args.epochs, max_ep_len=args.max_cycle,
        trained_dir=args.trained_dir, 
        recurrent=args.recurrent
        )
