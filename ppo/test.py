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
                    
def make_jobmatch_env(comm, **kwargs):
# def make_jobmatch_env(n_freelancers, n_recruiters, 
#                   max_cycles=500):
    env_1 = job_match_v1.parallel_env_1(**kwargs)
    env_2 = job_match_v1.parallel_env_2(**kwargs)
    env_3 = job_match_v1.parallel_env_3(**kwargs)
    env_1.reset()
    env_2.reset()
    env_3.reset()
    
    recruiters = ["recruiter_{}".format(i) for i in range(kwargs["n_recruiters"])]
    freelancers = ["freelancer_{}".format(i) for i in range(kwargs["n_freelancers"])]
    
    return env_1, env_2, env_3, recruiters, freelancers

def main(env_fn=None, actor_critic=MLPActorCritic, ac_kwargs=dict(), trained_dir=None, 
        beta=False, comm=False, dist_action=False, render=False, 
        recurrent=False, adv_train=False, ablate_kwargs=None, count_agent=None, 
        max_ep_len=200, epochs=20, num_agent=9, num_adv=0, 
        random_attacker=False, test_random = False
        ):

    
    print('-----Test Performance {} Epochs---'.format(epochs))
    
    # Instantiate environment
    env, good_agent_name, adv_agent_name = env_fn()
    observation_space = env.observation_spaces[good_agent_name[0]]
    action_space      = env.action_spaces[good_agent_name[0]]
    act_dim           = action_space.shape
    if comm:
        comm_space    = env.communication_spaces[good_agent_name[0]]
        if ablate_kwargs is None or ablate_kwargs['adv_train']:
            obs_dim       = (observation_space.shape[0]  + comm_space.shape[0],)
        else:
            num_agents = len(good_agent_name) + len(adv_agent_name)
            obs_dim       = (observation_space.shape[0]  + 
                             comm_space.shape[0]*ablate_kwargs['k']//(num_agents-1),)
    else:
        obs_dim = env.observation_spaces[good_agent_name[0]].shape 
    if beta:
        high = torch.from_numpy(action_space.high).to(Param.device).type(Param.dtype)
        low = torch.from_numpy(action_space.low).to(Param.device).type(Param.dtype)
    #print("Obs dim:{}, Act dim:{}".format(obs_dim[0], act_dim))
    
    # Create actor-critic module
    ac = actor_critic(obs_dim[0], action_space, **ac_kwargs).to(Param.device)
    if trained_dir is not None:
        state_dict, mean, std = torch.load(trained_dir, map_location=Param.device)
        ac.load_state_dict(state_dict)
        ac.moving_mean = mean
        ac.moving_std = std
    
    if random_attacker:
        ac = RandomAttacker(action_space)
    
    mean, _ = test_return(env, ac, epochs, max_ep_len, good_agent_name, 
                            dist_action, comm, recurrent, ablate_kwargs,
                            random_ac = test_random)
    print('-----[Number of Agents:{} Number of Adversary:{} Ablation Size:{} Median Sample Size {}] {} Epochs Performance:{}------'.\
          format(num_agent, num_adv, ablate_kwargs['k'], ablate_kwargs['median'], epochs, mean))
   
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action="store_true")
    parser.add_argument('--cuda', type=int, default=2)
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
    args = parser.parse_args()
    gym.logger.set_level(40)

    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.cuda)))
    
    env = make_jobmatch_env(
        n_freelancers=args.n_freelancers, 
        n_recruiters=args.n_recruiters,
        
    main(lambda:env, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, beta=args.beta, 
                       recurrent=args.recurrent, ep_len=args.max_cycle), 
        epochs=args.epochs, max_ep_len=args.max_cycle,
        beta=args.beta, comm=args.comm, dist_action=args.dist_action, trained_dir=args.trained_dir, 
        render=args.render, recurrent=args.recurrent, 
        adv_train=True if args.convert_adv else False,
        ablate_kwargs=ablate_kwargs, num_agent=args.n_pursuers, 
        num_adv = len(args.convert_adv) if args.convert_adv is not None else 0,
        random_attacker=args.test_random_attacker, test_random = args.test_random)
