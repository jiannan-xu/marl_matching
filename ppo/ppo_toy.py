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
import re     

def shuffle(comm, k):
    choice = np.random.choice(comm.shape[0], k, replace=False)
    return comm[choice]

def concatenate(obs, comm, ablate_kwargs=None):
    for agent in obs:
        if ablate_kwargs is None or agent in ablate_kwargs["adv_agents"]:
            obs[agent] = np.concatenate([obs[agent], comm[agent]])
        else:
            comm_agent = comm[agent].reshape(len(obs)-1, -1)
            comm_agent = shuffle(comm_agent, ablate_kwargs['k']).reshape(-1)
            obs[agent] = np.concatenate([obs[agent], comm_agent])
    return obs

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    obs_buf, act_buf, adv_buf, rew_buf, ret_buf, val_buf: numpy array
    logp_array: torch tensor
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf  = np.zeros(combined_shape(size, obs_dim))
        self.act_buf  = np.zeros(combined_shape(size, act_dim))
        self.adv_buf  = np.zeros(size)
        self.rew_buf  = np.zeros(size)
        self.ret_buf  = np.zeros(size)
        self.val_buf  = np.zeros(size)
        self.logp_buf = torch.zeros(size).to(Param.device).type(Param.dtype)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, hiddens=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf, 0), np.std(self.adv_buf, 0)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=from_numpy(self.obs_buf), act=from_numpy(self.act_buf), ret=from_numpy(self.ret_buf),
                    adv=from_numpy(self.adv_buf), logp=self.logp_buf)
        return data

def make_jobmatch_env(**kwargs):
    env_toy = job_match_v1.parallel_env_toy(**kwargs)
    env_toy.reset()
    agents = ["agent_{}".format(i) for i in range(kwargs["n_agents_freelancers"])]
    return env_toy, agents
               
def ppo(env_fn=None, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=1, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10, epoch_smoothed=10,
        no_save=False, verbose=False, log_freq=50, trained_dir=None, 
        obs_normalize=False, recurrent=False, data_dir=None
        ):
 
    # Setup logger file and reward file
    logger_file = open(os.path.join(data_dir, "logger_ppo.txt"), "a")
    rew_file = open(os.path.join(data_dir, "reward_ppo.txt"), "wt")
    
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env_toy, agents = env_fn()
    observation_space = env_toy.observation_spaces[agents[0]]
    action_space = env_toy.action_spaces[agents[0]]
    print('observation space: {}'.format(observation_space))
    print('action space: {}'.format(action_space))
    act_dim = action_space.shape
    obs_dim = spaces.utils.flatdim(observation_space)

    print("Obs dim:{}, Act dim:{}".format(obs_dim, act_dim))
    
    # Create actor-critic module
    ac_kwargs['beta'] = False
    ac_1 = actor_critic(obs_dim, action_space, **ac_kwargs).to(Param.device)

    if trained_dir is not None:
        print("loaded pretrained model from", trained_dir)
        ac_name = os.path.join(trained_dir, "ac.pt")
        
        state_dict, mean, std = torch.load(ac_name, map_location=Param.device)
        ac_1.load_state_dict(state_dict)
        ac_1.moving_mean = mean
        ac_1.moving_std = std
        ac_1.MovingMeanStd.old_m = mean
        ac_1.MovingMeanStd.old_s = std
        ac_1.MovingMeanStd.n     = 40000
    
    # Set up experience buffer
    bufs = {}
    for agent in agents:
        bufs[agent] = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, ac):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # print(obs.shape, act.shape, adv.shape, logp_old.shape)
        # print(obs, act, adv, logp_old)
        obs = ac.normalize(obs)
        act_info = dict(act_mean=torch.mean(act), act_std=torch.std(act))
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = clipped.type(Param.dtype).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info, act_info

    # Set up function for computing value loss
    def compute_loss_v(data, ac):
        obs, ret = data['obs'], data['ret']
        obs = ac.normalize(obs)
        return ((ac.v(obs) - ret)**2).mean()
        
    # Set up optimizers for policy and value function
    pi_optimizer_1 = Adam(ac_1.pi.parameters(), lr=pi_lr)
    vf_optimizer_1 = Adam(ac_1.v.parameters(), lr=vf_lr)

    def update(data, pi_optimizer, vf_optimizer, ac):
        pi_l_old, pi_info_old, act_info = compute_loss_pi(data, ac)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, ac).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info, act_info = compute_loss_pi(data, ac)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl and not verbose:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            pi_optimizer.step()
        stop_iter = i
        
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, ac)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        act_mean, act_std = act_info['act_mean'], act_info['act_std']
        return dict(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     StopIter=stop_iter,
                     Act_Mean=act_mean,
                     Act_Std=act_std)
    
    # Prepare for interaction with environment
    start_time = time.time()
    epoch_return, best_eval_return = [], -np.inf
    good_total_rewards = np.zeros(len(agents))
    best_eval_performance = -np.inf
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o1 = env.reset()
        ep_len, terminal = 0, False 
        avg_return, avg_len = [], []
        
        ### Initialize hidden states for reucrrent network
        if recurrent:
            ac_1.initialize()
            

        good_actions_1, values_1, log_probs_1 = {},{},{}
        for agent in agents:
            a, v1, logp = ac_1.step(torch.from_numpy(o1[agent]).\
                            to(Param.device).type(Param.dtype), train=obs_normalize)
            # print('1', agent, a, v1, logp)
            # logger_file.write('1 {} {} {} {}\n'.format(agent, a, v1, logp))
            values_1[agent] = v1
            log_probs_1[agent] = torch.mean(logp)
            a = a.cpu().numpy()
            good_actions_1[agent] = a
        # reward_1 is empty
        next_o1, reward_1, done_1, infos_1 = env.step(good_actions_1)
        

        for i in range(len(agents)):
            agent = agents[i]
            if agent in env.agents:
                good_total_rewards[i] += reward_1[agent]
            if done_1[agent]:
                terminal = True
            # print(o1[agent])
            # print(good_actions_1[agent])
            # print(reward_1[agent])
            # print(values_1[agent])
            # print(log_probs_1[agent])
            bufs[agent].store(o1[agent], good_actions_1[agent],
                                reward_1[agent], values_1[agent], log_probs_1[agent])
            
        for agent in agents:
            _, v1, _ = ac_1.step(torch.from_numpy(o1[agent]).to(Param.device).type(Param.dtype), 
                                train=obs_normalize)
            bufs[agent].finish_path(v1.item())
            
        ep_ret = good_total_rewards[i]
        good_total_rewards = np.zeros(len(good_total_rewards))        
        avg_return.append(ep_ret)
        avg_len.append(ep_len)
                
        
        
        # Perform PPO update!
        agents_data_1 = []
        for agent in agents:
            agents_data_1.append(bufs[agent].get())
            
        obs_buf_1  = torch.cat([data['obs'] for data in agents_data_1])
        act_buf_1  = torch.cat([data['act'] for data in agents_data_1])
        ret_buf_1  = torch.cat([data['ret'] for data in agents_data_1])
        adv_buf_1  = torch.cat([data['adv'] for data in agents_data_1])
        logp_buf_1 = torch.cat([data['logp'] for data in agents_data_1])
        
        data_1 = dict(obs=obs_buf_1, act=act_buf_1, ret=ret_buf_1,
                    adv=adv_buf_1, logp=logp_buf_1)

        
        param_dict_1 = update(data_1, pi_optimizer_1, vf_optimizer_1, ac_1)
        
        epoch_return.append(sum(avg_return)/len(avg_return))
            
        print("----------------------Epoch {}----------------------------".format(epoch))
        logger_file.write("----------------------Epoch {}----------------------------\n".format(epoch))
        print("EpRet:{}".format(sum(avg_return)/len(avg_return)))
        logger_file.write("EpRet:{}\n".format(sum(avg_return)/len(avg_return)))
        print("EpLen:{}".format(sum(avg_len)/len(avg_len)))
        logger_file.write("EpLen:{}\n".format(sum(avg_len)/len(avg_len)))
        print('V Values:{}'.format(v1))
        logger_file.write('V Values:{}\n'.format(v1))
        print('Total Interaction with Environment:{}'.format((epoch+1)*steps_per_epoch))
        logger_file.write('Total Interaction with Environment:{}\n'.format((epoch+1)*steps_per_epoch))
        print('LossPi:{}'.format(param_dict_1['LossPi']))
        logger_file.write('LossPi:{}\n'.format(param_dict_1['LossPi']))
        print('LossV:{}'.format(param_dict_1['LossV']))
        logger_file.write('LossV:{}\n'.format(param_dict_1['LossV']))
        print('DeltaLossPi:{}'.format(param_dict_1['DeltaLossPi']))
        logger_file.write('DeltaLossPi:{}\n'.format(param_dict_1['DeltaLossPi']))
        print('DeltaLossV:{}'.format(param_dict_1['DeltaLossV']))
        logger_file.write('DeltaLossV:{}\n'.format(param_dict_1['DeltaLossV']))
        print('Entropy:{}'.format(param_dict_1['Entropy']))
        logger_file.write('Entropy:{}\n'.format(param_dict_1['Entropy']))
        print('ClipFrac:{}'.format(param_dict_1['ClipFrac']))
        logger_file.write('ClipFrac:{}\n'.format(param_dict_1['ClipFrac']))
        print('KL:{}'.format(param_dict_1['KL']))
        logger_file.write('KL:{}\n'.format(param_dict_1['KL']))
        print('StopIter:{}'.format(param_dict_1['StopIter']))
        logger_file.write('StopIter:{}\n'.format(param_dict_1['StopIter']))
        print('Time:{}'.format(time.time()-start_time))
        logger_file.write('Time:{}\n'.format(time.time()-start_time))
        rew_file.write("Episode {}  EpRet:{}\n".format(epoch, sum(avg_return)/len(avg_return)))
        rew_file.flush()
        logger_file.flush()

        # print(env_3.aec_env.env.employment_status)
        # logger_file.write(str(env_3.aec_env.env.employment_status))
        
        ### Every few epochs, evaluate the current model's performance and save the model 
        ### if the eval performance is the best in the history.
        #if (log_freq is not None) and (epoch+1) % log_freq == 0:
            #performance,_ = test_return(env, ac, 50, 200, good_agent_name, dist_action, comm, recurrent, ablate_kwargs)
            #if performance > best_eval_performance:
                #best_eval_performance = performance
        if not no_save:
            ac_1.save(model_name='ac_1.pt')
                    
    logger_file.close()
    rew_file.close()
    
   

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--no-save', action="store_true")
    parser.add_argument('--no-cuda', action="store_true")
    parser.add_argument('--vf-lr', type=float, default=1e-3)
    parser.add_argument('--pi-lr', type=float, default=3e-4)
    parser.add_argument('--cuda', type=int, default=0) # default cuda:_
    parser.add_argument('--env', type=str, default='JobMatching') # check
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99) # ppo gamma
    parser.add_argument('--seed', '-s', type=int, default=0) 
    # parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epoch-smoothed', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--log-freq', type=int, default=50)
    parser.add_argument('--obs-normalize', action="store_true")
    
    parser.add_argument('--trained-dir', type=str, default="toy_models")
    parser.add_argument('--data-dir', type=str, default="toy_data")
    
    parser.add_argument('--n-freelancers', type=int, default=10)
    parser.add_argument('--n-recruiters', type=int, default=10)
    # parser.add_argument('--max-cycle', type=int, default=200)
    parser.add_argument('--recurrent', action="store_true")
    parser.add_argument('--exp-no', type=int, default=1)
    # parser.add
    
    args = parser.parse_args()
    gym.logger.set_level(40)
    
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.cuda)))
    
    budget, num_of_skills, pay_low, pay_high, rate_freelancer, rate_recruiter, base_price, u_ij, v_ij = get_parameters(args.n_freelancers, args.n_recruiters, args.exp_no)
    ### Please make sure the following argument names are the same as the FoodCollector's init function
    # def __init__(self, budget, num_of_skills, pay_low, pay_high, 
    #                 rate_freelancer, rate_recruiter, base_price, u_ij, v_ij, 
    #                 max_cycles, n_agents_recruiters = 2, 
    #                 n_agents_freelancers = 2, local_ratio = 1, **kwargs):
    env = make_jobmatch_env(base_price=base_price, u_ij=u_ij, v_ij=v_ij,
                            max_cycles=args.max_cycle, n_agents_freelancers=args.n_freelancers, n_agents_recruiters=args.n_recruiters)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    logger_file = open(os.path.join(args.data_dir, "logger_ppo.txt"), "wt")
    logger_file.write("Number of Freelancers: {}  Number of Recruiters:{}\n".\
                      format(args.n_freelancers, args.n_recruiters))
    logger_file.close()
        
    
    ppo(lambda:env, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, recurrent=args.recurrent, ep_len=args.max_cycle), 
        gamma=args.gamma, vf_lr=args.vf_lr, pi_lr=args.pi_lr,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=args.max_cycle,
        epoch_smoothed=args.epoch_smoothed,
        no_save = args.no_save, verbose=args.verbose, log_freq = args.log_freq, 
        obs_normalize = args.obs_normalize, 
        trained_dir=args.trained_dir, recurrent=args.recurrent, data_dir=args.data_dir)