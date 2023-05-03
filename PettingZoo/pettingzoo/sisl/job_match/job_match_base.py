import os

from torch import poisson
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces,MultiBinary
from gym.utils import seeding
from .._utils import Agent
import pygame
import math


class Individual(Agent):
    def __init__(self, idx, budget, pay_low, pay_upper, base_price, n_skills, group, n_agent_recruiters, n_agent_freelancers):
        '''
        idx: index of the agent
        group: group of the agent (recruiters or freelancers)
        group_num: number of agents in the group (recruiters or freelancers)
        '''
        self._idx = idx
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

        # set edges to be None
        self._edges = None
        
        # recruiter specific actions
        # set post price to be None
        self._price_post =  None
        # set updated price to be None
        self._price_post_update = None
        # set offer decision to be None
        self._offer_decision = None

        # freelancer specific actions
        # set application decision to be None
        self._submit_app = None
        # set bid price to be None
        self._price_bid = None
        # set offer decision to be None
        self._accept_offer = None

        if group == 'recruiters':
            self._edges = np.zeros(n_agent_freelancers)
        elif group == 'freelancers':
            self._edges = np.zeros(n_agent_recruiters)
        else:
            raise ValueError("Invalid group")
    
    @property
    def budget(self):
        if self._budget is None and self._group == 'recruiters':
            raise ValueError("Budget is not set")
        else:
            return self._budget
        
    @property
    def base_price(self):
        if self._base_price is None and self._group == 'recruiters':
            raise ValueError("Base price is not set")
        else:
            return self._base_price
        
    @property
    def n_skills(self):
        if self._n_skills is None and self._group == 'freelancers':
            raise ValueError("Number of skills is not set")
        else:
            return self._n_skills
        
    @property
    def pay_low(self):
        if self._pay_low is None and self._group == 'freelancers':
            raise ValueError("Pay low is not set")
        else:
            return self._pay_low
        
    @property
    def pay_upper(self):
        if self._pay_upper is None and self._group == 'freelancers':
            raise ValueError("Pay upper is not set")
        else:
            return self._pay_upper

    @property
    def observation_space(self):
        # let the observation space be the edges of the bipartite graph?
        if self.group == 'recruiters':
            return MultiBinary(self.n_agent_freelancers)
        elif self.group == 'freelancers':
            return MultiBinary(self.n_agent_recruiters)
        else:
            raise ValueError("Invalid observation space")
    
    @property
    def action_space(self):
        if self.group == 'recruiters':
            # 1. set base price
            # Do we need to make the base price an action for recruiters?
            self._price_post  = spaces.Box(low=0, high=self._budget, shape=(1,))

            # 2. screen applications
            self._offer_decision = spaces.Discrete(2) # 0: reject, 1: accept

            # 3. update base price
            self._price_post_update = spaces.Box(low=0, high=self._budget, shape=(1,))

            return self._price_post, self._offer_decision, self._price_post_update
        
        elif self.group == 'freelancers':
            # submit application or not
            self._submit_app = spaces.Discrete(2) # 0: not submit, 1: submit

            # bid price
            self._price_bid = spaces.Box(low=self._pay_low, high=self._pay_upper, shape=(1,))

            # accept the offer or not
            self._accept_offer = spaces.Discrete(2) # 0: not accept, 1: accept

            return self._submit_app, self._price_bid, self._accept_offer
        else:
            raise ValueError("Invalid action space")


class Jobmatching():
    """A Bipartite Networked Multi-agent Environment."""

    def __init__(self, budget, match_reward,max_cycles, 
                 use_groudtruth=False, n_agents_recruiters = 2, n_agents_freelancers = 2, local_ratio = 1, **kwargs):
        '''
        n_agents_recruiters: Number of agents in group A (recruiters)
        n_agents_freelancers: Number of agents in group B (freelancers)
        match_reward: reward for matching
        use_groudtruth: whether to use groundtruth for matching
        max_cycles: After max_cycles steps all agents will return done
        '''
        self.n_agents_recruiters = n_agents_recruiters
        self.n_agents_freelancers = n_agents_freelancers
        self.budget = budget # need to specify the exact budget for recruiters
        self.match_reward = match_reward # need to specify the exact reward for matching
        self.use_groudtruth = use_groudtruth
        self.local_ratio = local_ratio
        self.seed()
        self._recruiters = [
            Individual(recruiters_idx + 1, budget = self.budget[recruiters_idx], group='recruiters',)
            for recruiters_idx in range(self.n_agents_recruiters)
        ]

        self._freelancers = [
            Individual(freelancers_idx + 1, budget = None, group='freelancers',)
            for freelancers_idx in range(self.n_agents_freelancers)
        ]

        self.action_space_recruiters = [agent.action_space for agent in self._recruiters]
        self.action_space_freelancers = [agent.action_space for agent in self._freelancers]
        self.observation_space_recruiters = [
            agent.observation_space for agent in self._recruiters]
        self.observation_space_freelancers = [
            agent.observation_space for agent in self._freelancers]
        print("observation space recruiters", self.observation_space_recruiters)
        print("observation space freelancers", self.observation_space_freelancers)
        self.max_cycles = max_cycles
        self.step_count = 0
        self.reset()

    # close the game window
    def close(self):
        if self.renderOn:
            # pygame.event.pump()
            pygame.display.quit()
            pygame.quit()

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


    def reset(self):
        # self.frames = 0

        # # Initialize recruiters and freelancers
        # for recruiter in self._recruiters:
        #     recruiter.set_position(self._generate_coord(recruiter._radius))

        # for freelancer in self._freelancers:
        #     freelancer.set_position(self._generate_coord(freelancer._radius))
        
        # initialize the observation for recruiters and freelancers
        obslist_recruiters,obslist_freelancers = self.observe_list(self)

        # combine the observation for recruiters and freelancers
        obs_list = obslist_recruiters + obslist_freelancers
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_recruiters + self.n_agents_freelancers)]
        self.control_rewards = [0 for _ in range(self.n_agents_recruiters + self.n_agents_freelancers)]
        self.last_dones = [False for _ in range(self.n_agents_recruiters + self.n_agents_freelancers)]
        self.last_obs = obs_list
        return obs_list[0]

    def observe_list(self):
        obslist_recruiters = []
        obslist_freelancers = []
        for agent_idx in range(self.n_agent_recruiters):
            one_hot_recruiters = np.zeros(self.n_agent_recruiters)
            one_hot_recruiters[agent_idx] = 1.0
            base_price = self._recruiters[agent_idx].base_price
            budget = self._recruiters[agent_idx].budget
            obslist_recruiters.append(
                np.concatenate([
                one_hot_recruiters, base_price, budget
                ]))
        for agent_idx in range(self.n_agent_freelancers):
            one_hot_freelancers = np.zeros(self.n_agent_freelancers)
            one_hot_freelancers[agent_idx] = 1.0
            n_skills = self._freelancers[agent_idx].n_skills
            obslist_freelancers.append(one_hot_freelancers, n_skills)
        return obslist_recruiters, obslist_freelancers

    # # TODO: Pending editing
    def convert_action(self, action):
        '''convert a discrete action to continuous acceleration'''
        action_map = np.array([[0,0], [1, 0], [0, 1], [-1, 0], [0, -1],
                            [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5],[-0.5, -0.5]])
        return action_map[action] * 0.05 # *self.pursuer_max_accel*0.5

    def step(self, action, agent_id, group, is_last):
        if self.dist_action:
            action = self.convert_action(action)
        else:
            action = np.asarray(action)
            # action = action.reshape(2) # reshape the action to be a 2D array
        if group == 'recruiters':
            r = self._recruiters[agent_id]
        if group == 'freelancers':
            f = self._freelancers[agent_id]
        
        # TODO: Pending editing (Define control rewards)
    #    # Average thrust penalty among all agents, and assign each agent global portion designated by (1 - local_ratio)
    #     self.control_rewards = (accel_penalty / self.n_pursuers) * np.ones(self.n_pursuers) * (1 - self.local_ratio)
    #     # Assign the current agent the local portion designated by local_ratio
    #     self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        if is_last:
            rewards = np.zeros(self.n_pursuers)
            obs_list = self.observe_list(self)
            self.last_obs = obs_list

            # TODO: Update the reward function
            # # new reward: 
            # for g in range(self.n_groups):
            #     visibles = [e.visible for e in self._evaders[g*self.n_evaders_pergroup:(g+1)*self.n_evaders_pergroup]]
            #     rewards[g] -= np.array(visibles).sum() * self.food_alive_penalty

            self.local_reward = rewards
            self.global_reward = self.local_reward.mean()
            # Distribute local and global rewards according to local_ratio
            self.last_rewards = self.local_reward * self.local_ratio + self.global_reward * (1 - self.local_ratio)
        return self.observe(agent_id)

    def observe(self, agent):
        return np.array(self.last_obs[agent], dtype=np.float32)

    #TODO: Pending editing
    def render(self, mode="human"):
        if not self.renderOn:
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.window_size*self.pixel_scale, self.window_size*self.pixel_scale))
            else:
                self.screen = pygame.Surface((self.window_size*self.pixel_scale, self.window_size*self.pixel_scale))
            self.renderOn = True

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if mode == "human":
            pygame.display.flip()
        return np.transpose(new_observation, axes=(1, 0, 2)) if mode == "rgb_array" else None
