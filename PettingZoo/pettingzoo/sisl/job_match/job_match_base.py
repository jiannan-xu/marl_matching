import os

from torch import poisson
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces
from gym.utils import seeding
from .._utils import Agent
import pygame
import math

MAX_BASE_PRICE = 20
MIN_BASE_PRICE = 1
DEFAULT_BASE_PRICE = 10
NOT_HIRED_PENALTY = 5
NOT_ENOUGH_EMPLOYERS_PENALTY = 5
MAX_NUM_SKILLS = 10
MIN_NUM_SKILLS = 1
MAX_RATE = 100
MIN_RATE = 1
DEFAULT_RATE = 50
MAX_BUDGET = 60

class Individual(Agent):
    def __init__(self, idx, rate, budget, pay_low, pay_upper, base_price, n_skills, group, n_agent_recruiters, n_agent_freelancers):
        '''
        idx: index of the agent
        group: group of the agent (recruiters or freelancers)
        group_num: number of agents in the group (recruiters or freelancers)
        '''
        self._idx = idx
        self._rate = rate
        # recruiter specific properties
        self._budget = budget # budget of the agent
        self._base_price = base_price # base price of the agent
        # freelancer specific properties
        self._n_skills = n_skills # num of skills
        self._pay_low = pay_low # lower bound of the payrange
        self._pay_upper = pay_upper # higher bound of the payrange
        # claim the agent to be in a group (recruiters or freelancers)
        self._group = group
        # claim the number of agents in the group (recruiters or freelancers)
        self.n_agent_recruiters = n_agent_recruiters
        self.n_agent_freelancers = n_agent_freelancers
        # step 1
        self.past_rate = np.zeros(self.n_agent_recruiters)
        self.past_offer_price = np.zeros(self.n_agent_recruiters)
        # step 2
        self.low_prices = np.zeros(self.n_agent_freelancers)
        self.high_prices = np.zeros(self.n_agent_freelancers)
        
        # if self.group == 'recruiters':
        #     self.has_application = np.zeros(self.n_agent_freelancers)
        #     self.has_offer = np.zeros(self.n_agent_freelancers)
        # elif self.group == 'freelancers':
        #     self.has_application = np.zeros(self.n_agent_recruiters)
        #     self.has_offer = np.zeros(self.n_agent_recruiters)
        # else:
        #     raise ValueError("Invalid group")

        # set edges to be None
        # self._edges = None
        
        # # recruiter specific actions
        # # set post price to be None
        # self._price_post =  None
        # # set updated price to be None
        # self._price_post_update = None
        # # set offer decision to be None
        # self._offer_decision = None

        # # freelancer specific actions
        # # set application decision to be None
        # self._submit_app = None
        # # set bid price to be None
        # self._price_bid = None
        # # set offer decision to be None
        # self._accept_offer = None

        # if group == 'recruiters':
        #     self._edges = np.zeros(n_agent_freelancers)
        # elif group == 'freelancers':
        #     self._edges = np.zeros(n_agent_recruiters)
        # else:
        #     raise ValueError("Invalid group")
    
    @property
    def observation_space_1(self):
        return spaces.Dict({
            'base_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),
            'rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_recruiters,)),
            'past_offer_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),
            'employment_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),
        })
        
    @property
    def observation_space_2(self):
        return spaces.Dict({
            'num_of_skills': spaces.Box(low=0, high=MAX_NUM_SKILLS, shape=(self.n_agent_freelancers,)),
            'pay_low': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),
            'pay_high': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),
            'rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_freelancers,)),
            'offer_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),
            'employment_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),
        })
        
    @property
    def observation_space_3(self):
        return spaces.Dict({
            'budget': spaces.Box(low=0, high=MAX_BUDGET, shape=(self.n_agent_recruiters,)),
            'base_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),
            'offer_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),
            'rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_recruiters,)),
            'offer_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),
            'employment_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),
        })
    
    @property
    def action_space_1(self):
        return spaces.MultiBinary(self.n_agent_recruiters)
    
    @property
    def action_space_2(self):
        return spaces.Box(low=self.low_prices, high=self.high_prices)
    
    @property
    def action_space_3(self):
        return spaces.Discrete(self.n_agent_recruiters + 1)
    
    # @property
    # def rate(self):
    #     assert self._rate is not None, "Rate is not set"
    #     return self._rate
    
    # @property
    # def budget(self):
    #     if self._budget is None and self._group == 'recruiters':
    #         raise ValueError("Budget is not set")
    #     else:
    #         return self._budget
        
    # @property
    # def base_price(self):
    #     if self._base_price is None and self._group == 'recruiters':
    #         raise ValueError("Base price is not set")
    #     else:
    #         return self._base_price
        
    # @property
    # def n_skills(self):
    #     if self._n_skills is None and self._group == 'freelancers':
    #         raise ValueError("Number of skills is not set")
    #     else:
    #         return self._n_skills
        
    # @property
    # def pay_low(self):
    #     if self._pay_low is None and self._group == 'freelancers':
    #         raise ValueError("Pay low is not set")
    #     else:
    #         return self._pay_low
        
    # @property
    # def pay_upper(self):
    #     if self._pay_upper is None and self._group == 'freelancers':
    #         raise ValueError("Pay upper is not set")
    #     else:
    #         return self._pay_upper

    def reset(self):
        self.past_rate = np.zeros(self.n_agent_recruiters)
        self.past_offer_price = np.zeros(self.n_agent_recruiters)
        self.low_prices = np.zeros(self.n_agent_freelancers)
        self.high_prices = np.zeros(self.n_agent_freelancers)
        # if self.group == 'recruiters':
        #     self.has_application = np.zeros(self.n_agent_freelancers)
        #     self.has_offer = np.zeros(self.n_agent_freelancers)
        # elif self.group == 'freelancers':
        #     self.has_application = np.zeros(self.n_agent_recruiters)
        #     self.has_offer = np.zeros(self.n_agent_recruiters)
        # else:
        #     raise ValueError("Invalid group")
   
class Freelancer_1(Agent):
    def __init__(self, idx, rate, pay_low, pay_upper, n_skills, n_agent_recruiters, n_agent_freelancers):
        self.idx = idx
        self.rate = rate
        self.n_skills = n_skills # num of skills
        self.pay_low = pay_low # lower bound of the payrange
        self.pay_upper = pay_upper # higher bound of the payrange
        # claim the number of agents in the group (recruiters or freelancers)
        self.n_agent_recruiters = n_agent_recruiters
        self.n_agent_freelancers = n_agent_freelancers
        # # environment variables
        # self.past_rate = np.zeros(self.n_agent_recruiters)
        # self.past_offer_price = np.zeros(self.n_agent_recruiters)
    
    @property
    def observation_space(self):
        return spaces.Dict({
            # 'budget': spaces.Box(low=0, high=MAX_BUDGET, shape=(self.n_agent_recruiters,)),                               # recruiter's budget; not available to freelancers
            'base_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),            # recruiter's base price
            'num_of_skills': spaces.Box(low=MIN_NUM_SKILLS, high=MAX_NUM_SKILLS, shape=(self.n_agent_recruiters,)),         # recruiter's number of skills
            'offer_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),           # recruiter's past offer price, if any
            'low_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),            # freelancer's low price
            'high_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),           # freelancer's high price
            'freelancer_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_recruiters,)),                   # freelancer's rate
            'recruiter_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_recruiters,)),                    # recruiter's past rate, if any
            'employment_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),                   # recruiter's employment status
        }) # R * (4 + F)
        
    @property
    def action_space(self):
        return spaces.MultiBinary(self.n_agent_recruiters) # R

    # def reset(self):
    #     self.past_rate = np.zeros(self.n_agent_recruiters)
    #     self.past_offer_price = np.zeros(self.n_agent_recruiters)
   
class Recruiter_2(Agent):
    def __init__(self, idx, rate, budget, base_price, pay_low, pay_high, n_agent_recruiters, n_agent_freelancers):
        '''
        idx: index of the agent
        group: group of the agent (recruiters or freelancers)
        group_num: number of agents in the group (recruiters or freelancers)
        '''
        self.idx = idx
        self.rate = rate
        # recruiter specific properties
        self.budget = budget # budget of the agent
        self.base_price = base_price # base price of the agent
        # claim the number of agents in the group (recruiters or freelancers)
        self.n_agent_recruiters = n_agent_recruiters
        self.n_agent_freelancers = n_agent_freelancers
        
        self.pay_low = pay_low
        self.pay_high = pay_high
        # # step 2
        # self.low_prices = np.zeros(self.n_agent_freelancers)
        # self.high_prices = np.zeros(self.n_agent_freelancers)
        
    @property
    def observation_space(self):
        return spaces.Dict({
            'base_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),    # recruiter's base price
            'num_of_skills': spaces.Box(low=0, high=MAX_NUM_SKILLS, shape=(self.n_agent_freelancers,)),             # freelancer's number of skills
            'pay_low': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),      # freelancer's lower bound of the payrange
            'pay_high': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),     # freelancer's higher bound of the payrange
            'budget': spaces.Box(low=0, high=MAX_BUDGET, shape=(self.n_agent_recruiters,)),                         # recruiter's budget; not available to freelancers
            'freelancer_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_freelancers,)),          # freelancer's rate
            'recruiter_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_freelancers,)),           # recruiter's rate
            'offer_status': spaces.MultiBinary([self.n_agent_freelancers, self.n_agent_recruiters]),                # freelancer's offer status
            'employment_status': spaces.MultiBinary([self.n_agent_freelancers, self.n_agent_recruiters]),           # freelancer's employment status
        }) # F * (5 + 2*R)

    @property
    def action_space(self):
        # print(self.pay_low, self.pay_high, self.n_agent_freelancers)
        return spaces.Box(low=np.zeros(self.n_agent_freelancers), high=self.pay_high, shape=(self.n_agent_freelancers,)) # F
    
    # def reset(self):
    #     self.low_prices = np.zeros(self.n_agent_freelancers)
    #     self.high_prices = np.zeros(self.n_agent_freelancers)
     
class Freelancer_3(Agent):
    def __init__(self, idx, rate, pay_low, pay_upper, n_skills, n_agent_recruiters, n_agent_freelancers):
        self.idx = idx
        self.rate = rate
        self.n_skills = n_skills # num of skills
        self.pay_low = pay_low # lower bound of the payrange
        self.pay_upper = pay_upper # higher bound of the payrange
        # claim the number of agents in the group (recruiters or freelancers)
        self.n_agent_recruiters = n_agent_recruiters
        self.n_agent_freelancers = n_agent_freelancers
        # # step 1
        # self.past_rate = np.zeros(self.n_agent_recruiters)
        # self.past_offer_price = np.zeros(self.n_agent_recruiters)
    
    @property
    def observation_space(self):
        return spaces.Dict({
            'base_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),  # recruiter's base price
            'num_of_skills': spaces.Box(low=0, high=MAX_NUM_SKILLS, shape=(self.n_agent_freelancers,)),             # freelancer's number of skills
            'pay_low': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),      # freelancer's lower bound of the payrange
            'pay_high': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),     # freelancer's higher bound of the payrange
            'offer_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)), # recruiter's past offer price, if any
            'freelancer_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_recruiters,)),         # freelancer's rate
            'recruiter_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_recruiters,)),          # recruiter's past rate, if any
            'offer_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),
            'employment_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),         # recruiter's employment status
        }) # R * (5 + 2*F)
        
    @property
    def action_space(self):
        return spaces.Discrete(self.n_agent_recruiters + 1) # R

    # def reset(self): 
    #     self.past_rate = np.zeros(self.n_agent_recruiters)
    #     self.past_offer_price = np.zeros(self.n_agent_recruiters)
                        
class Jobmatching():
    """A Bipartite Networked Multi-agent Environment."""

    def __init__(self, budget, num_of_skills, pay_low, pay_high, rate_freelancer, rate_recruiter, base_price, u_ij, v_ij, max_cycles, 
                 n_agents_recruiters = 2, n_agents_freelancers = 2, local_ratio = 1, **kwargs):
        '''
        n_agents_recruiters: Number of agents in group A (recruiters)
        n_agents_freelancers: Number of agents in group B (freelancers)
        match_reward: reward for matching
        local_ratio: Proportion of reward allocated locally vs distributed globally among all agents
        max_cycles: After max_cycles steps all agents will return done
        '''
        self.n_agents_recruiters = n_agents_recruiters
        self.n_agents_freelancers = n_agents_freelancers
        self.seed()
        # class Individual(Agent):
        #     def __init__(self, idx, budget, pay_low, pay_upper, base_price, n_skills, group, n_agent_recruiters, n_agent_freelancers):
        self._recruiters = [
            Individual(idx + 1, rate_recruiter[idx], budget[idx], None, None, base_price[idx], None, 'recruiters', self.n_agents_recruiters, self.n_agents_freelancers)
            for idx in range(self.n_agents_recruiters)
        ]

        self._freelancers = [
            Individual(idx + 1, rate_freelancer[idx], None, pay_low[idx], pay_high[idx], None, num_of_skills[idx], 'freelancers', self.n_agents_recruiters, self.n_agents_freelancers)
            for idx in range(self.n_agents_freelancers)
        ]
        
        self.employment_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.application_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.offer_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        
        self.u_ij = np.array(u_ij)
        self.v_ij = np.array(v_ij)
        
        self.action_space_1 = [agent.action_space_1 for agent in self.n_agents_freelancers]
        self.action_space_2 = [agent.action_space_2 for agent in self.n_agents_recruiters]
        self.action_space_3 = [agent.action_space_3 for agent in self.n_agents_freelancers]
        
        self.observation_space_1 = [agent.observation_space_1 for agent in self.n_agents_freelancers]
        self.observation_space_2 = [agent.observation_space_2 for agent in self.n_agents_recruiters]
        self.observation_space_3 = [agent.observation_space_3 for agent in self.n_agents_freelancers]
        
        self.last_rewards_freelancer = [np.float64(0) for _ in range(self.n_agents_freelancers)]
        # self.control_rewards_freelancer = [0 for _ in range(self.n_agents_freelancers)]
        self.last_dones_freelancer = [False for _ in range(self.n_agents_freelancers)]

        self.last_rewards_recruiter = [np.float64(0) for _ in range(self.n_agents_recruiters)]
        # self.control_rewards_recruieter = [0 for _ in range(self.n_agents_recruiters)]
        self.last_dones_recruiter = [False for _ in range(self.n_agents_recruiters)]
        
        self.last_obs_1 = [None for _ in range(self.n_agents_freelancers)]
        self.last_obs_2 = [None for _ in range(self.n_agents_recruiters)]
        self.last_obs_3 = [None for _ in range(self.n_agents_freelancers)]
        
        self.max_cycles = max_cycles
        self.local_ratio = local_ratio
        self.step_count = 0
        self.reset()

    @property
    def agents_recruiters(self):
        return self._recruiters

    @property
    def agents_freelancers(self):
        return self._freelancers

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observe_list_1(self):
        obs_list_1 = []
        for i in range(self.n_agents_freelancers):
            obs_i = []
            for idx in range(self.n_agents_recruiters):
                rate = self._freelancers[i].past_rate[idx]
                if rate == 0:
                    rate = DEFAULT_RATE
                offer_price = self._freelancers[i].past_offer_price[idx]
                if offer_price == 0:
                    offer_price = self._recruiters[idx].base_price
                obs_i.append(np.concatenate([self._recruiters[idx].base_price, rate, offer_price, self.employment_status[:, idx].ravel()]))
            obs_i = np.array(obs_i)
            obs_list_1.append(obs_i)
        return obs_list_1
    
    def observe_list_2(self):
        obs_list_2 = []
        for i in range(self.n_agents_recruiters):
            obs_i = []
            for idx in range(self.n_agents_freelancers):
                if self.application_status[idx, i] == 1:
                    obs_i.append(np.concatenate([self._freelancers[idx].n_skills, self._freelancers[idx].pay_low, 
                                                self._freelancers[idx].pay_high, self._freelancers[idx].rate,
                                                self.offer_status[idx, :].ravel(), self.employment_status[idx, :].ravel()]))
                else:
                    obs_i.append(np.zeros(2 * self.n_agents_recruiters + 4))
            obs_i = np.array(obs_i)
            obs_list_2.append(obs_i)
        return obs_list_2
    
    def observe_list_3(self):
        obs_list_3 = []
        for i in range(self.n_agents_freelancers):
            obs_i = []
            for idx in range(self.n_agents_recruiters):
                if self.application_status[i, idx] == 1:
                    obs_i.append(np.concatenate([self._recruiters[idx].budget, self._recruiters[idx].base_price,
                                                self._recruiters[idx].offer_price, self._recruiters[idx].rate,
                                                self.offer_status[:, idx].ravel(), self.employment_status[:, idx].ravel()]))
                else:
                    obs_i.append(np.zeros(2 * self.n_agents_freelancers + 4))
            obs_i = np.array(obs_i)
            obs_list_3.append(obs_i)
        return obs_list_3
        
    def reset(self):
        self.step_count = 0
        self.employment_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.application_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.offer_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        for recruiter in self._recruiters:
            recruiter.reset()
        for freelancer in self._freelancers:
            freelancer.reset()
        self.last_rewards_freelancer = [np.float64(0) for _ in range(self.n_agents_freelancers)]
        # self.control_rewards_freelancer = [0 for _ in range(self.n_agents_freelancers)]
        self.last_dones_freelancer = [False for _ in range(self.n_agents_freelancers)]

        self.last_rewards_recruiter = [np.float64(0) for _ in range(self.n_agents_recruiters)]
        # self.control_rewards_recruieter = [0 for _ in range(self.n_agents_recruiters)]
        self.last_dones_recruiter = [False for _ in range(self.n_agents_recruiters)]
        self.last_obs_1 = self.observe_list_1()
        self.last_obs_2 = self.observe_list_2()
        self.last_obs_3 = self.observe_list_3()

    def step1(self, action, agent_id, is_last):
        """
        1. Update application status in the environment
        2. The freelancer will update the past rate of the recruiter
        3. The recruiter will update the upper and lower price of the freelancer
        """
        action = np.asarray(action)
        for i in range(self.n_agents_recruiters):
            if action[i] == 1:
                self.application_status[agent_id, i] = 1
                self._freelancers[agent_id].past_rate[i] = self._recruiters[i].rate
                self._recruiters[i].low_prices[agent_id] = self._freelancers[agent_id].pay_low
                self._recruiters[i].high_prices[agent_id] = self._freelancers[agent_id].pay_high
        if is_last:
            self.last_obs_2 = self.observe_list_2()
        
    def step2(self, action, agent_id, is_last):
        """
        1. Update offer status in the environment
        2. The freelancer will update the past offer price of the recruiter
        """
        action = np.asarray(action)
        nonzeros = np.count_nonzero(action)
        employed = np.count_nonzero(self.employment_status[:, agent_id])
        budget = self._recruiters[agent_id].budget
        if nonzeros <= budget - employed:
            for i in range(self.n_agents_freelancers):
                if action[i] > 0:
                    self.offer_status[i, agent_id] = 1
                    self._freelancers[i].past_offer_price[agent_id] = action[i]
        else:
            # choose the top budget - employed freelancers with the highest offer price
            price = np.partition(action, -budget + employed)[-budget + employed]
            for i in range(self.n_agents_freelancers):
                if action[i] >= price:
                    self.offer_status[i, agent_id] = 1
                    self._freelancers[i].past_offer_price[agent_id] = action[i]
        if is_last:
            self.last_obs_3 = self.observe_list_3()
        
    def step3(self, action, agent_id, is_last):
        """
        1. Update employment status in the environment
        """
        # nonzero offer and have applied
        if action != 0 and self.application_status[agent_id, action - 1] == 1:
            self.employment_status[agent_id, action - 1] = 1
        if is_last:
            # update rewards
            # rewards_freelancer = np.zeros(self.n_agents_freelancers)
            # rewards_recruiter = np.zeros(self.n_agents_recruiters)
            self.last_obs_1 = self.observe_list_1()
            rewards_freelancer, rewards_recruiter = self.rewards_handler()
            local_rewards_freelancer = rewards_freelancer
            local_rewards_recruiter = rewards_recruiter
            global_rewards_freelancer = local_rewards_freelancer.mean()
            global_rewards_recruiter = local_rewards_recruiter.mean()
            self.last_rewards_freelancer = local_rewards_freelancer * self.local_ratio + global_rewards_freelancer * (1 - self.local_ratio)
            self.last_rewards_recruiter = local_rewards_recruiter * self.local_ratio + global_rewards_recruiter * (1 - self.local_ratio)
            self.step_count += 1

    def rewards_handler(self):
        rewards_freelancer = np.zeros(self.n_agents_freelancers)
        rewards_recruiter = np.zeros(self.n_agents_recruiters)
        employed_freelancers = np.zeros(self.n_agents_freelancers)
        employer_num = np.zeros(self.n_agents_recruiters)
        for i in range(self.n_agents_freelancers):
            for j in range(self.n_agents_recruiters):
                if self.employment_status[i, j] == 1:
                    employed_freelancers[i] = 1
                    employer_num[j] += 1
                    rewards_freelancer[i] = self._freelancers[i].past_offer_price[j] - self.v_ij[i, j] 
                    rewards_recruiter[j] += self.u_ij[i, j] - self._freelancers[i].past_offer_price[j]
        for i in range(self.n_agents_freelancers):
            if employed_freelancers[i] == 0:
                rewards_freelancer[i] = -NOT_HIRED_PENALTY
        for j in range(self.n_agents_recruiters):
            rewards_recruiter[j] -= (self._recruiters[j].budget - employer_num[j]) * NOT_HIRED_PENALTY

    # def step(self, action, agent_id, group, is_last):

    def observe1(self, agent):
        return np.array(self.last_obs_1[agent], dtype=np.float32)
    
    def observe2(self, agent):
        return np.array(self.last_obs_2[agent], dtype=np.float32)
    
    def observe3(self, agent):
        return np.array(self.last_obs_3[agent], dtype=np.float32)

class Jobmatching_step1():
    """A Bipartite Networked Multi-agent Environment."""
    def __init__(self, budget, num_of_skills, pay_low, pay_high, rate_freelancer, rate_recruiter, base_price, u_ij, v_ij, max_cycles, 
                 n_agents_recruiters, n_agents_freelancers, local_ratio = 1, **kwargs):
        self.n_agents_recruiters = n_agents_recruiters
        self.n_agents_freelancers = n_agents_freelancers
        self.num_agents = n_agents_freelancers
        self.seed()
        self.agents = [Freelancer_1(idx + 1, rate_freelancer[idx], pay_low[idx], pay_high[idx], num_of_skills[idx], self.n_agents_recruiters, self.n_agents_freelancers) for idx in range(self.n_agents_freelancers)]
        
        self.num_of_skills = num_of_skills
        self.budget = budget
        self.pay_low = pay_low
        self.pay_high = pay_high
        self.rate_freelancer = rate_freelancer
        self.rate_recruiter = rate_recruiter
        self.base_price = base_price
        self.new_actions = dict()
        
        self.u_ij = np.array(u_ij)
        self.v_ij = np.array(v_ij)
        
        self.action_space = [agent.action_space for agent in self.agents]        
        self.observation_space = [agent.observation_space for agent in self.agents]
        # environment parameters
        self.employment_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))    # employment status (used at every step, updated at 3rd step)
        self.application_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))   # application status (used at 2nd, 3rd step, updated at 1st step)
        self.applied_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))       # applied status (used at 1st step, updated at 1st step)
        self.offer_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))         # offer status (used at 3rd step, updated at 2nd step)
        self.past_offer_price = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))     # past offer price (used at 1st step, updated at 2nd step)
        
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_freelancers)]
        self.last_dones = [False for _ in range(self.n_agents_freelancers)]
        self.last_obs = [None for _ in range(self.n_agents_freelancers)]
        
        self.max_cycles = max_cycles
        self.local_ratio = local_ratio
        self.step_count = 0
        self.reset()

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def observe_list(self):
        obs_list_1 = []
        for i in range(self.n_agents_freelancers):
            obs_i = []
            freelancer_rate = self.agents[i].rate
            for idx in range(self.n_agents_recruiters):
                if self.applied_status[i, idx] == 1:
                    recruiter_rate = self.rate_recruiter[idx]
                    offer_price = self.past_offer_price[i, idx]
                else:
                    recruiter_rate = DEFAULT_RATE
                    offer_price = self.base_price[idx]
                obs_i.append(np.append([self.base_price[idx], self.num_of_skills[idx], offer_price, self.pay_low[idx], self.pay_high[idx], freelancer_rate, recruiter_rate], self.employment_status[:, idx].ravel()))
            obs_i = np.array(obs_i).flatten()
            obs_list_1.append(obs_i)
        return obs_list_1
        
    def reset(self):
        self.step_count = 0
        
        self.employment_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.application_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.applied_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.offer_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.past_offer_price = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_freelancers)]
        self.last_dones = [False for _ in range(self.n_agents_freelancers)]
        self.last_obs = self.observe_list()
        return self.last_obs[0]

    def step(self, action, agent_id, is_last):
        """
        1. Update application status in the environment
        2. The freelancer will update the past rate of the recruiter
        3. The recruiter will update the upper and lower price of the freelancer
        """
        action = np.asarray(action)
        # for i in range(self.n_agents_recruiters):
        #     if action[i] == 1:
        #         self.application_status[agent_id, i] = 1
        #         self.applied_status[agent_id, i] = 1
        #     else:
        #         self.application_status[agent_id, i] = 0
        for i in range(self.n_agents_recruiters):
           self.application_status[agent_id, i] = 1
           self.applied_status[agent_id, i] = 1
        self.new_actions[agent_id] = action
        if is_last:
            self.last_obs = self.observe_list()
            self.step_count += 1

    def update_reward_step(self, last_rewards):
        self.last_rewards = last_rewards

    def observe(self, agent):
        return np.array(self.last_obs[agent], dtype=np.float32)

    def update_env_2(self, offer_status, past_offer_price):
        self.offer_status = offer_status
        self.past_offer_price = past_offer_price
        
    def update_env_3(self, employment_status):
        self.employment_status = employment_status

    def output_for_update_env(self):
        return self.application_status

class Jobmatching_step2():
    """A Bipartite Networked Multi-agent Environment."""
    # def __init__(self, budget, num_of_skills, pay_low, pay_high, rate_freelancer, rate_recruiter, base_price, u_ij, v_ij, max_cycles, 
    #              n_agents_recruiters = 2, n_agents_freelancers = 2, local_ratio = 1, **kwargs):
    def __init__(self, budget, num_of_skills, pay_low, pay_high, rate_freelancer, rate_recruiter, base_price, u_ij, v_ij, max_cycles, 
                 n_agents_recruiters, n_agents_freelancers, local_ratio = 1, **kwargs):
        # print(n_agents_freelancers, n_agents_recruiters)
        self.n_agents_recruiters = n_agents_recruiters
        self.n_agents_freelancers = n_agents_freelancers
        self.num_agents = n_agents_recruiters
        self.seed()
        # __init__(self, idx, rate, budget, base_price, n_agent_recruiters, n_agent_freelancers)
        self.agents = [Recruiter_2(idx + 1, rate_recruiter[idx], budget[idx], base_price[idx], pay_low, pay_high, self.n_agents_recruiters, self.n_agents_freelancers)
            for idx in range(self.n_agents_recruiters)]
        
        self.budget = budget
        self.pay_low = pay_low
        self.pay_high = pay_high
        self.rate_freelancer = rate_freelancer
        self.rate_recruiter = rate_recruiter
        self.base_price = base_price
        self.num_of_skills = num_of_skills
        self.new_actions = dict()
        
        self.action_space = [agent.action_space for agent in self.agents]        
        self.observation_space = [agent.observation_space for agent in self.agents]
        # environment parameters
        self.employment_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))    # employment status (used at every step, updated at 3rd step)
        self.application_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))   # application status (used at 2nd, 3rd step, updated at 1st step)
        self.applied_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))       # applied status (used at 1st step, updated at 1st step)
        self.offer_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))         # offer status (used at 3rd step, updated at 2nd step)
        self.past_offer_price = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))     # past offer price (used at 1st step, updated at 2nd step)
        
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_recruiters)]
        self.last_dones = [False for _ in range(self.n_agents_recruiters)]
        self.last_obs = [None for _ in range(self.n_agents_recruiters)]
        
        self.max_cycles = max_cycles
        self.local_ratio = local_ratio
        self.step_count = 0
        self.reset()

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observe_list(self):
        obs_list_1 = []
        for i in range(self.n_agents_recruiters):
            obs_i = []
            # 'num_of_skills': spaces.Box(low=0, high=MAX_NUM_SKILLS, shape=(self.n_agent_freelancers,)),             # freelancer's number of skills
            # 'pay_low': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),      # freelancer's lower bound of the payrange
            # 'pay_high': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_freelancers,)),     # freelancer's higher bound of the payrange
            # 'freelancer_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_freelancers,)),          # freelancer's rate
            # 'recruiter_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_freelancers,)),           # recruiter's rate
            # 'offer_status': spaces.MultiBinary([self.n_agent_freelancers, self.n_agent_recruiters]),                # freelancer's offer status
            # 'employment_status': spaces.MultiBinary([self.n_agent_freelancers, self.n_agent_recruiters]),           # freelancer's employment status
            for idx in range(self.n_agents_freelancers):
                if self.application_status[idx, i] == 1:
                    num_of_skills = self.num_of_skills[idx]
                    pay_low = self.pay_low[idx]
                    pay_high = self.pay_high[idx]
                    freelancer_rate = self.rate_freelancer[idx]
                else:
                    num_of_skills = 0
                    pay_low = 0
                    pay_high = 0
                    freelancer_rate = 0
                recruiter_rate = self.rate_recruiter[i]
                offer_status = self.offer_status[idx].ravel()
                employment_status = self.employment_status[idx].ravel()
                base_price = self.base_price[idx]
                budget = self.budget[i]
                obs_i.append(np.concatenate((np.array([base_price, num_of_skills, pay_low, pay_high, budget, freelancer_rate, recruiter_rate]), offer_status, employment_status)))
            obs_i = np.array(obs_i).flatten()
            obs_list_1.append(obs_i)
        return obs_list_1
        
    def reset(self):
        self.step_count = 0
        
        self.employment_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.application_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.applied_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.offer_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.past_offer_price = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_recruiters)]
        self.last_dones = [False for _ in range(self.n_agents_recruiters)]
        self.last_obs = self.observe_list()
        
        return self.last_obs[0]

    def step(self, action, agent_id, is_last):
        """
        1. Update offer status in the environment
        2. The freelancer will update the past offer price of the recruiter
        """

        action = np.asarray(action)

        '''
        if bg - employed >= nonzeros:
            for i in range(self.n_agents_freelancers):
                if self.application_status[i, agent_id] == 1:
                    self.offer_status[i, agent_id] = 1
                    self.past_offer_price[i, agent_id] = action[i]
                    new_action[i] = action[i]
                else:
                    self.offer_status[i, agent_id] = 0
                    self.past_offer_price[i, agent_id] = self.pay_low[i]
                    new_action[i] = self.pay_low[i]
        else:
            # sort the action
            candidate_prices = []
            for i in range(self.n_agents_freelancers):
                if self.application_status[i, agent_id] == 1:
                    candidate_prices.append(action[i])
                else:
                    candidate_prices.append(self.pay_low[i])
            candidate_prices = np.array(candidate_prices)
            k = int(-bg + employed)
            if k == 0:
                for i in range(self.n_agents_freelancers):
                    if self.employment_status[i, agent_id] == 1:
                        self.offer_status[i, agent_id] = 1
                        self.past_offer_price[i, agent_id] = action[i]
                        new_action[i] = action[i]
                    else:
                        self.offer_status[i, agent_id] = 0
                        self.past_offer_price[i, agent_id] = self.pay_low[i]
                        new_action[i] = self.pay_low[i]
            else:
                price = np.partition(candidate_prices, k)[k]
                for i in range(self.n_agents_freelancers):
                    if action[i] >= price and self.application_status[i, agent_id] == 1:
                        self.offer_status[i, agent_id] = 1
                        self.past_offer_price[i, agent_id] = action[i]
                        new_action[i] = action[i]
                    else:
                        self.offer_status[i, agent_id] = 0
                        self.past_offer_price[i, agent_id] = self.pay_low[i]
                        new_action[i] = self.pay_low[i]
        '''
        bg = self.budget[agent_id]
        new_action = np.zeros(self.n_agents_freelancers)

        total_employed_cost = 0
        for i in range(self.n_agents_freelancers):
            if self.application_status[i, agent_id] == 1:
                total_employed_cost += action[i]

        if bg >= total_employed_cost:
            for i in range(self.n_agents_freelancers):
                if self.application_status[i, agent_id] == 1:
                    self.offer_status[i, agent_id] = 1
                    self.past_offer_price[i, agent_id] = action[i]
                    new_action[i] = action[i]
                else:
                    self.offer_status[i, agent_id] = 0
                    self.past_offer_price[i, agent_id] = 0
                    new_action[i] = 0
        else:
            for i in range(self.n_agents_freelancers):
                self.offer_status[i, agent_id] = 0
                self.past_offer_price[i, agent_id] = 0
            # sort the action
            total_employed_cost = 0
            budget_left = bg
            sorted_action = np.flip(np.sort(action))
            # print("sorted_action: ", sorted_action)
            for act in sorted_action:
                if budget_left == 0:
                    break
                if act > budget_left:
                    for i in range(self.n_agents_freelancers):
                        if act == action[i] and budget_left > 0:
                            self.offer_status[i, agent_id] = 1
                            self.past_offer_price[i, agent_id] = budget_left
                            new_action[i] = budget_left
                            budget_left = 0
                    break
                else:
                    for i in range(self.n_agents_freelancers):
                        if act == action[i] and budget_left >= act:
                            self.offer_status[i, agent_id] = 1
                            self.past_offer_price[i, agent_id] = act
                            new_action[i] = act
                            budget_left -= act
                        elif act == action[i] and budget_left < act:
                            self.off_status[i, agent_id] = 1
                            self.past_offer_price[i, agent_id] = budget_left
                            new_action[i] = budget_left
                            budget_left = 0
        
        self.new_actions[agent_id] = new_action


        # print("action: ", action)
        # print("new action: ", new_action)
        # print("budget: ", budget)
        # assert 0

        # if budget == employed:
        #     for i in range(self.n_agents_freelancers):
        #         if self.application_status[i, agent_id] == 1:
        #             self.past_offer_price[i, agent_id] = action[i]
        # else:
        #     for i in range(self.n_agents_freelancers):
        #         self.offer_status[i, agent_id] = 0
        #         self.past_offer_price[i, agent_id] = 0
        #     if nonzeros <= budget - employed:
        #         for i in range(self.n_agents_freelancers):
        #             if self.application_status[i, agent_id] == 1:
        #                 self.offer_status[i, agent_id] = 1
        #                 self.past_offer_price[i, agent_id] = action[i]
        #             else:
        #                 self.offer_status[i, agent_id] = 0
        #                 self.past_offer_price[i, agent_id] = 0
        #     else:
        #         # choose the top budget - employed freelancers with the highest offer price
        #         candidate_prices = []
        #         for i in range(self.n_agents_freelancers):
        #             if self.application_status[i, agent_id] == 1:
        #                 candidate_prices.append(action[i])
        #             else:
        #                 candidate_prices.append(0)
        #         candidate_prices = np.array(candidate_prices)
        #         k = int(-budget + employed)
        #         # print(k)
        #         price = np.partition(candidate_prices, k)[k]
        #         for i in range(self.n_agents_freelancers):
        #             if action[i] >= price and self.application_status[i, agent_id] == 1:
        #                 self.offer_status[i, agent_id] = 1
        #                 self.past_offer_price[i, agent_id] = action[i]
        #             else:
        #                 self.offer_status[i, agent_id] = 0
        #                 self.past_offer_price[i, agent_id] = 0

        # print("offer status: {}".format(self.offer_status))
        # print("offer price: {}".format(self.past_offer_price))

        if is_last:
            self.last_obs = self.observe_list()
            self.step_count += 1

    def update_reward_step(self, last_rewards):
        self.last_rewards = last_rewards

    def observe(self, agent):
        return np.array(self.last_obs[agent], dtype=np.float32)

    def update_env_1(self, application_status):
        self.application_status = application_status
        
    def update_env_3(self, employment_status):
        self.employment_status = employment_status
        
    def output_for_update_env(self):
        return self.offer_status, self.past_offer_price

class Jobmatching_step3():
    """A Bipartite Networked Multi-agent Environment."""
    def __init__(self, budget, num_of_skills, pay_low, pay_high, rate_freelancer, rate_recruiter, base_price, u_ij, v_ij, max_cycles, 
                 n_agents_recruiters, n_agents_freelancers, local_ratio = 1, **kwargs):
        self.n_agents_recruiters = n_agents_recruiters
        self.n_agents_freelancers = n_agents_freelancers
        self.num_agents = n_agents_freelancers
        self.seed()
        self.agents = [Freelancer_3(idx + 1, rate_freelancer[idx], pay_low[idx], pay_high[idx], num_of_skills[idx], self.n_agents_recruiters, self.n_agents_freelancers) for idx in range(self.n_agents_freelancers)]
        
        self.num_of_skills = num_of_skills
        self.budget = budget
        self.pay_low = pay_low
        self.pay_high = pay_high
        self.rate_freelancer = rate_freelancer
        self.rate_recruiter = rate_recruiter
        self.base_price = base_price
        self.new_actions = dict()
        
        self.u_ij = np.array(u_ij)
        self.v_ij = np.array(v_ij)
        
        self.action_space = [agent.action_space for agent in self.agents]        
        self.observation_space = [agent.observation_space for agent in self.agents]
        # environment parameters
        self.employment_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))    # employment status (used at every step, updated at 3rd step)
        self.application_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))   # application status (used at 2nd, 3rd step, updated at 1st step)
        self.applied_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))       # applied status (used at 1st step, updated at 1st step)
        self.offer_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))         # offer status (used at 3rd step, updated at 2nd step)
        self.past_offer_price = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))     # past offer price (used at 1st step, updated at 2nd step)
        
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_freelancers)]
        self.last_recruiter_rewards = [np.float64(0) for _ in range(self.n_agents_recruiters)]
        self.last_dones = [False for _ in range(self.n_agents_freelancers)]
        self.last_obs = [None for _ in range(self.n_agents_freelancers)]
        
        self.max_cycles = max_cycles
        self.local_ratio = local_ratio
        self.step_count = 0
        self.reset()

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def observe_list(self):
        # 'budget': spaces.Box(low=0, high=MAX_BUDGET, shape=(self.n_agent_recruiters,)),                     # recruiter's budget; not available to freelancers
        # 'base_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)),  # recruiter's base price
        # 'offer_price': spaces.Box(low=MIN_BASE_PRICE, high=MAX_BASE_PRICE, shape=(self.n_agent_recruiters,)), # recruiter's past offer price, if any
        # 'freelancer_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_recruiters,)),         # freelancer's rate
        # 'recruiter_rate': spaces.Box(low=MIN_RATE, high=MAX_RATE, shape=(self.n_agent_recruiters,)),          # recruiter's past rate, if any
        # 'offer_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),
        # 'employment_status': spaces.MultiBinary([self.n_agent_recruiters, self.n_agent_freelancers]),         # recruiter's employment status
        obs_list = []
        for i in range(self.n_agents_freelancers):
            obs_i = []
            for idx in range(self.n_agents_recruiters):
                budget = self.budget[idx]
                base_price = self.base_price[idx]
                rate_freelancer = self.rate_freelancer[i]
                rate_recruiter = self.rate_recruiter[idx]
                if self.offer_status[i, idx] == 1:
                    num_of_skills = self.num_of_skills[idx]
                    pay_low = self.pay_low[idx]
                    pay_high = self.pay_high[idx]
                    past_offer_price = self.past_offer_price[i, idx]
                else:
                    num_of_skills = 0
                    pay_low = 0
                    pay_high = 0
                    past_offer_price = 0
                obs_i.append(np.concatenate((np.array([base_price, num_of_skills, pay_low, pay_high, past_offer_price, rate_freelancer, rate_recruiter]), self.offer_status[:, idx].ravel(), self.employment_status[:, idx].ravel())))
            obs_i = np.array(obs_i).flatten()
            obs_list.append(obs_i)
        return obs_list
          
    def reset(self):
        self.step_count = 0
        
        self.employment_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.application_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.applied_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.offer_status = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        self.past_offer_price = np.zeros((self.n_agents_freelancers, self.n_agents_recruiters))
        
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_freelancers)]
        self.last_recruiter_rewards = [np.float64(0) for _ in range(self.n_agents_recruiters)]
        self.last_dones = [False for _ in range(self.n_agents_freelancers)]
        self.last_obs = self.observe_list()
        
        return self.last_obs[0]

    def rewards_handler(self):
        rewards_freelancer = np.zeros(self.n_agents_freelancers)
        rewards_recruiter = np.zeros(self.n_agents_recruiters)
        employed_freelancers = np.zeros(self.n_agents_freelancers)
        employer_num = np.zeros(self.n_agents_recruiters)
        for i in range(self.n_agents_freelancers):
            for j in range(self.n_agents_recruiters):
                if self.employment_status[i, j] == 1:
                    employed_freelancers[i] = 1
                    employer_num[j] += 1
                    rewards_freelancer[i] = self.past_offer_price[i, j] - self.v_ij[i, j] 
                    rewards_recruiter[j] += self.u_ij[i, j] - self.past_offer_price[i, j]
        for i in range(self.n_agents_freelancers):
            if employed_freelancers[i] == 0:
                rewards_freelancer[i] = -NOT_HIRED_PENALTY
        for j in range(self.n_agents_recruiters):
            rewards_recruiter[j] -= int(self.budget[j]/DEFAULT_BASE_PRICE - employer_num[j]) * NOT_HIRED_PENALTY
        return rewards_freelancer, rewards_recruiter

    def step(self, action, agent_id, is_last):
        """
        1. Update employment status in the environment
        """
        # nonzero offer and have applied
        for i in range(self.n_agents_recruiters):
            self.employment_status[agent_id, i] = 0
        if action == self.n_agents_recruiters:
            self.new_actions[agent_id] = self.n_agents_recruiters
        elif self.application_status[agent_id, action] == 1 and self.offer_status[agent_id, action] == 1:
            self.employment_status[agent_id, action] = 1
            self.new_actions[agent_id] = action
        else:
            self.new_actions[agent_id] = self.n_agents_recruiters

        if is_last:
            # update rewards
            # rewards_freelancer = np.zeros(self.n_agents_freelancers)
            # rewards_recruiter = np.zeros(self.n_agents_recruiters)
            self.last_obs = self.observe_list()
            rewards_freelancer, rewards_recruiter = self.rewards_handler()
            local_rewards_freelancer = rewards_freelancer
            local_rewards_recruiter = rewards_recruiter
            global_rewards_freelancer = local_rewards_freelancer.mean()
            global_rewards_recruiter = local_rewards_recruiter.mean()
            last_rewards_freelancer = local_rewards_freelancer * self.local_ratio + global_rewards_freelancer * (1 - self.local_ratio)
            last_rewards_recruiter = local_rewards_recruiter * self.local_ratio + global_rewards_recruiter * (1 - self.local_ratio)
            self.last_rewards = last_rewards_freelancer
            self.last_recruiter_rewards = last_rewards_recruiter
            # print("self last rewards freelancer", self.last_rewards)
            # print("self last rewards recruiter", self.last_recruiter_rewards)
            self.step_count += 1

    def observe(self, agent):
        return np.array(self.last_obs[agent], dtype=np.float32)

    def update_env_2(self, offer_status, past_offer_price):
        self.offer_status = offer_status
        self.past_offer_price = past_offer_price

    def update_env_1(self, application_status):
        self.application_status = application_status
        
    def output_for_update_env(self):
        return self.employment_status
    
    def output_rewards(self):
        return self.last_rewards, self.last_recruiter_rewards # freelancers, recruiters

class ToyAgent(Agent):
    def __init__(self, idx, n_agent_recruiters):
        self.idx = idx
        self.n_agent_recruiters = n_agent_recruiters

    def observation_space(self):
        return spaces.Discrete(self.n_agent_recruiters + 1)

    def action_space(self):
        return spaces.Discrete(self.n_agent_recruiters)

class ToyExample():
    def __init__(self, base_price, n_agents_freelancers, n_agents_recruiters, **kwargs):
        self.n_agents_freelancers = n_agents_freelancers
        self.n_agents_recruiters = n_agents_recruiters
        self.seed()
        self.base_price = base_price

        self.num_employees = np.zeros(self.n_agents_recruiters)
        self.previous_employer = np.array([-1 for _ in range(self.n_agents_freelancers)])

        self.agents = [ToyAgent(i, self.n_agents_recruiters) for i in range(self.n_agents_freelancers)]

        self.action_space = [agent.action_space for agent in self.agents]        
        self.observation_space = [agent.observation_space for agent in self.agents]
        
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_freelancers)]
        self.last_dones = [False for _ in range(self.n_agents_freelancers)]
        self.last_obs = [None for _ in range(self.n_agents_freelancers)]

        self.step_count = 0
        self.reset()

    def observe_list(self):
        obs_list = []
        for i in range(self.n_agents_freelancers):
            obs_list.append(self.num_employees[i])
        return obs_list

    def reset(self):
        self.num_employees = np.zeros(self.n_agents_recruiters)
        self.previous_employer = np.array([-1 for _ in range(self.n_agents_freelancers)])
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_freelancers)]
        self.last_dones = [False for _ in range(self.n_agents_freelancers)]
        self.last_obs = self.observe_list()
        return self.last_obs[0]

    def rewards_handler(self):
        rewards = np.zeros(self.n_agents_recruiters)
        for i in range(self.n_agents_recruiters):
            rewards[i] = self.base_price[self.previous_employer[i]]
        return rewards

    def step(self, action, agent_id, is_last):
        if self.previous_employer[agent_id] != -1:
            self.num_employees[self.previous_employer[agent_id]] -= 1
            self.num_employees[action] += 1
            self.previous_employer[agent_id] = action
        else:
            self.num_employees[action] += 1
            self.previous_employer[agent_id] = action

        if is_last:
            self.last_obs = self.observe_list()
            rewards = self.rewards_handler()
            local_rewards = rewards
            global_rewards = local_rewards.mean()
            last_rewards = local_rewards * self.local_ratio + global_rewards * (1 - self.local_ratio)
            self.last_rewards = last_rewards
            self.step_count += 1

    def observe(self, agent):
        return np.array(self.last_obs[agent], dtype=np.float32)