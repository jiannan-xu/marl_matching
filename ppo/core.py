from param import Param
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete, MultiBinary
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.nn import GRU, Linear
import copy


# Return all combinations of size k of numbers
def makeCombi(n, k):
    def makeCombiUtil(n, left, k, tmp, idx):
        if (k == 0):
            idx.append(random.sample(copy.deepcopy(tmp), len(tmp)))
            return
        for i in range(left, n + 1):
            tmp.append(i-1)
            makeCombiUtil(n, i + 1, k - 1, tmp, idx)
            tmp.pop()
    tmp, idx = [], []
    makeCombiUtil(n, 1, k, tmp, idx)
    return idx

def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)

    
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

### Clip the Action
def clip(action, low, high):
    return np.clip(action, low, high)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    
class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, recurrent=False, ep_len=1000):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPMultiBinaryActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, recurrent=False, ep_len=1000):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Binomial(logits=logits)
        # return torch.sigmoid(logits)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
        # return torch.sum(act*torch.log(pi) + (1-act)*torch.log(1-pi), dim=1)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, 
                 recurrent=False, ep_len=1000):
        super().__init__()
        self.obs_dim   = obs_dim
        self.recurrent = recurrent
        self.ep_len    = ep_len
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        if not recurrent:
            self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        else:
            self.num_layers = len(hidden_sizes)
            self.hidden_size= hidden_sizes[0]
            self.mu_gru  = GRU(obs_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True)
            self.mu_net  = Linear(hidden_sizes[0], act_dim)
    
    ### Initialize latent state, called only when recurrent is True
    def initialize(self):
        self.h = torch.zeros(self.num_layers, 1, self.hidden_size).to(Param.device).type(Param.dtype)
    
    def _distribution(self, obs, mu_h=None):
        std = torch.exp(self.log_std)
        if not self.recurrent:
            mu = self.mu_net(obs)
        else:
            if len(obs.shape) == 1:
                obs            = obs.reshape(1,1,-1)
                mu, self.h = self.mu_gru(obs, self.h)
                mu       = mu.reshape(-1)
                mu       = self.mu_net(mu)
            else:
                batch_size = obs.shape[0]//self.ep_len
                obs = obs.reshape(batch_size, self.ep_len, -1)
                h   = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(Param.device).type(Param.dtype)
                mu, _ = self.mu_gru(obs, h)
                mu   = mu.reshape(-1, self.hidden_size)
                mu   = self.mu_net(mu)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
class MLPBetaActor(Actor):
    
    ### Beta distribution, dealing with the case where action is bounded in the 
    ### box (-epsilon, epsilon)
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, high, recurrent=False, ep_len=1000):
        super().__init__()
        self.high = high
        self.ep_len = ep_len
        self.recurrent = recurrent
        self.obs_dim   = obs_dim
        if not recurrent:
            self.alpha = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
            self.beta  = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        else:
            self.num_layers = len(hidden_sizes)
            self.h_size = (self.num_layers, 1, hidden_sizes[0])
            self.hidden_size= hidden_sizes[0]
            self.alpha_gru  = GRU(obs_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True)
            self.beta_gru   = GRU(obs_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True)
            self.alpha      = Linear(hidden_sizes[0], act_dim)
            self.beta       = Linear(hidden_sizes[0], act_dim)
    
    ### Initialize latent state, called only when recurrent is True
    def initialize(self):
        self.alpha_h = torch.zeros(self.num_layers, 1, self.hidden_size).to(Param.device).type(Param.dtype)
        self.beta_h  = torch.zeros(self.num_layers, 1, self.hidden_size).to(Param.device).type(Param.dtype)
        
    ### Input shape: (obs_dim,) or (batch_size, obs_dim)
    def _distribution(self, obs):
        if not self.recurrent:
            alpha = self.alpha(obs)
            beta  = self.beta(obs)
        else:   
            if len(obs.shape) == 1:
                obs            = obs.reshape(1,1,-1)
                alpha, self.alpha_h = self.alpha_gru(obs, self.alpha_h)
                beta,  self.beta_h  = self.beta_gru(obs,  self.beta_h)
                alpha, beta    = alpha.reshape(-1), beta.reshape(-1)
            else:
                batch_size = obs.shape[0]//self.ep_len
                obs = obs.reshape(batch_size, self.ep_len, -1)
                alpha_h = torch.zeros(self.num_layers, batch_size, self.hidden_size).\
                        to(Param.device).type(Param.dtype)
                beta_h  = torch.zeros(self.num_layers, batch_size, self.hidden_size).\
                        to(Param.device).type(Param.dtype)
                alpha, _ = self.alpha_gru(obs, alpha_h)
                beta,  _  = self.beta_gru(obs, beta_h)
                alpha, beta    = alpha.reshape(-1, self.hidden_size), beta.reshape(-1, self.hidden_size)
            alpha = self.alpha(alpha)
            beta  = self.beta(beta)
        alpha = torch.log(1+torch.exp(alpha))+1
        beta  = torch.log(1+torch.exp(beta))+1
        return Beta(alpha, beta)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)   
    
class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, recurrent=False, ep_len=1000):
        super().__init__()
        self.ep_len = ep_len
        self.recurrent = recurrent
        if recurrent:
            self.num_layers = len(hidden_sizes)
            self.hidden_size= hidden_sizes[0]
            self.v_gru  = GRU(obs_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True)
            self.v      = Linear(hidden_sizes[0], 1)
        else:
            self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    
    ### Initialize latent state, called only when recurrent is True
    def initialize(self):
        self.h = torch.zeros(self.num_layers, 1, self.hidden_size).to(Param.device).type(Param.dtype)
    
    def forward(self, obs, v_h=None):
        if not self.recurrent:
            return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape
        else:
            if len(obs.shape) == 1:
                obs            = obs.reshape(1,1,-1)
                v, self.h = self.v_gru(obs, self.h)
                v       = v.reshape(-1)
            else:
                batch_size = obs.shape[0]//self.ep_len
                obs = obs.reshape(batch_size, self.ep_len, -1)
                h   = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(Param.device).type(Param.dtype)
                v, _ = self.v_gru(obs, h)
                v   = v.reshape(-1, self.hidden_size)
            return torch.squeeze(self.v(v), -1)
    
    
class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, action_space, hidden_sizes=(64,64), activation=nn.Tanh, beta=False, recurrent=False, ep_len=1000):
        super().__init__()
        
        self.beta = beta ### Whether to use beta distribution to deal with clipped action space
        self.recurrent = recurrent
        # policy builder depends on action space
        if isinstance(action_space, Box) and not beta:
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, 
                                       activation, recurrent, ep_len)
        elif isinstance(action_space, Discrete):
            self.beta = False
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, 
                                          activation, recurrent, ep_len)
        elif isinstance(action_space, MultiBinary):
            self.pi = MLPMultiBinaryActor(obs_dim, action_space.shape[0], hidden_sizes,
                                          activation, recurrent, ep_len)
        else:
            self.high = torch.from_numpy(action_space.high).type(Param.dtype).to(Param.device)
            self.low = torch.from_numpy(action_space.low).type(Param.dtype).to(Param.device)
            self.pi = MLPBetaActor(obs_dim, action_space.shape[0], 
                                   hidden_sizes, activation, self.high, 
                                   recurrent, ep_len)
        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation, recurrent, ep_len)
        
        self.MovingMeanStd = MovingMeanStd((obs_dim,))
        self.moving_mean = torch.zeros(obs_dim).to(Param.device).type(Param.dtype)
        self.moving_std  = torch.ones(obs_dim).to(Param.device).type(Param.dtype)
    
    def initialize(self):
        self.v.initialize()
        self.pi.initialize()
        
    def step(self, obs, train=False):
        with torch.no_grad():
            if train:
                self.MovingMeanStd.push(obs)
                self.moving_mean = self.MovingMeanStd.mean()
                self.moving_std  = self.MovingMeanStd.std()
            obs = (obs - self.moving_mean)/(self.moving_std+1e-6)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            if self.beta:
                a = a*(self.high-self.low)+self.low ### Clip to the correct range
            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]
        
    def save(self, log_dir='./learned_models/', model_name='ppo_policy'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        torch.save([self.state_dict(), self.moving_mean, self.moving_std], os.path.join(log_dir,model_name))
    
    ### Return Normalized Observation
    def normalize(self, obs):
        return (obs - self.moving_mean)/(self.moving_std+1e-6)
    
class RandomAttacker(nn.Module):

    def __init__(self, action_space):
        super().__init__()
        self.act_dim  = action_space.shape
        self.act_high = from_numpy(action_space.high)
        self.act_low  = from_numpy(action_space.low)
        
    def act(self, obs=None):
        act = torch.randint(high=2, size=self.act_dim).to(Param.device)
        return torch.where(act==0, self.act_low, self.act_high)
    
def shuffle(comm, k):
    choice = np.random.choice(comm.shape[0], k, replace=False)
    return comm[choice]


def concatenate(obs, comm, ablate_kwargs=None, idx_list=None):
    for agent in obs:
        if ablate_kwargs is None or agent in ablate_kwargs["adv_agents"]:
            obs[agent] = np.concatenate([obs[agent], comm[agent]])
        else:
            choice = np.random.choice(idx_list.shape[0], ablate_kwargs['median'], 
                                     replace=False)
            idxs = [idx_list[i] for i in choice]
            o  = np.stack([obs[agent]]*ablate_kwargs['median'])
            comm_agent  = comm[agent].reshape(len(obs)-1, -1)
            comm_agent  = np.concatenate([comm_agent[idx].reshape(1,-1) for idx in idxs])
            obs[agent]  = np.concatenate([o, comm_agent], axis=-1)
    return obs
## Train adversary
## default: benign agents -> shuffle

def test_return(env_1, env_2, env_3, ac_1, ac_2, ac_3, epochs, max_ep_len, 
                freelancers, recruiters, recurrent=False):
    logger_file = open(os.path.join("./data", "logger_test_ppo.txt"), "a")
    default_recruiter_name = recruiters[0]
    default_freelancer_name = freelancers[0]   
    action_space_1 = env_1.action_spaces[default_freelancer_name]
    action_space_2 = env_2.action_spaces[default_recruiter_name]
    action_space_3 = env_3.action_spaces[default_freelancer_name]
    low, high = action_space_2.low, action_space_2.high
        
    freelancer_rewards = [[]]*len(freelancers)
    recruiter_rewards  = [[]]*len(recruiters)

    for eps in range(epochs):
        freelancer_episode_rewards = np.zeros(len(freelancers))
        recruiter_episode_rewards  = np.zeros(len(recruiters))
        if recurrent:
            ac_1.initialize()
            ac_2.initialize()
            ac_3.initialize()

        o_1 = env_1.reset()
        o_2 = env_2.reset()
        o_3 = env_3.reset()
        done = False
        for t in range (max_ep_len):
            good_actions_1 = {}
            good_actions_2 = {}
            good_actions_3 = {}
            
            for agent in freelancers:
                a = ac_1.act(torch.from_numpy(o_1[agent]).to(Param.device).type(Param.dtype))
                a = a.cpu().numpy()
                good_actions_1[agent] = a
            next_o1, reward_1, done_1, _ = env_1.step(good_actions_1)
            app_status = env_1.aec_env.env.output_for_update_env()
            env_2.aec_env.env.update_env_1(app_status)
            env_3.aec_env.env.update_env_1(app_status)
            
            for agent in recruiters:
                a = ac_2.act(torch.from_numpy(o_2[agent]).to(Param.device).type(Param.dtype))
                a = a.cpu().numpy()
                a = np.clip(a, low, high)
                good_actions_2[agent] = a
            next_o2, reward_2, done_2, _ = env_2.step(good_actions_2)
            off_status, pri_status = env_2.aec_env.env.output_for_update_env()
            env_1.aec_env.env.update_env_2(off_status, pri_status)
            env_3.aec_env.env.update_env_2(off_status, pri_status)
            
            for agent in freelancers:
                a = ac_3.act(torch.from_numpy(o_3[agent]).to(Param.device).type(Param.dtype))
                a = a.item()
                good_actions_3[agent] = a
                
            next_o3, reward_3, done_3, _ = env_3.step(good_actions_3)
            emp_status = env_3.aec_env.env.output_for_update_env()
            env_1.aec_env.env.update_env_3(emp_status)
            env_2.aec_env.env.update_env_3(emp_status)
            
            _, r2 = env_3.aec_env.env.output_rewards()
            for i in range(len(recruiters)):
                reward_2[recruiters[i]] = r2[i]
            reward_1 = reward_3
            for i in range(len(freelancers)):
                agent = freelancers[i]
                if done_1[agent] or done_3[agent]:
                    done = True
                freelancer_episode_rewards[i] += reward_3[agent]
            for i in range(len(recruiters)):
                agent = recruiters[i]
                if done_2[agent]:
                    done = True
                recruiter_episode_rewards[i] += reward_2[agent]
            logger_file.write("action_1: {}\n".format(good_actions_1))
            logger_file.write("action_2: {}\n".format(good_actions_2))
            logger_file.write("action_3: {}\n".format(good_actions_3))
            logger_file.write("employment: {}\n".format(emp_status))
            o_1 = next_o1
            o_2 = next_o2
            o_3 = next_o3
            if done:
                for i in range(len(freelancers)):
                    freelancer_rewards[i].append(freelancer_episode_rewards[i])
                for i in range(len(recruiters)):
                    recruiter_rewards[i].append(recruiter_episode_rewards[i])
                break
    reward_f = np.array(freelancer_rewards)
    reward_r = np.array(recruiter_rewards)
    return np.mean(reward_f), np.std(reward_f), np.mean(reward_r), np.std(reward_r)


def get_parameters(n_freelancers, n_recruiters, exp_no):
    if exp_no == 0:
        # Toy example
        assert n_freelancers == 3
        assert n_recruiters == 3
        budget = np.ones(3)
        base_price = np.array([8, 12, 16])
        low_price = np.array([6, 10, 14])
        high_price = np.array([10, 14, 18])
        rate_freelancer = np.array([1, 50, 100])
        rate_recruiter = np.array([1, 50, 100])
        num_of_skills = np.array([1, 2, 3])
        u_ij = np.array([[8, 8, 8], [13, 13, 13], [18, 18, 18]])
        v_ij = np.array([[8, 8, 8], [11, 11, 11], [14, 14, 14]])
    else:
        return NotImplementedError
    return budget, base_price, low_price, high_price, rate_freelancer, rate_recruiter, num_of_skills, u_ij, v_ij

### Calculating moving meana and standard deviation
class MovingMeanStd:

    def __init__(self, shape):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.shape = shape

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        if self.n > 0:
            return self.new_m
        else:
            return torch.zeros(self.shape).to(Param.device).type(Param.dtype)

    def variance(self):
        if self.n > 1:
            return self.new_s / (self.n - 1) 
        else:
            return torch.ones(self.shape).to(Param.device).type(Param.dtype)

    def std(self):
        return torch.sqrt(self.variance())
