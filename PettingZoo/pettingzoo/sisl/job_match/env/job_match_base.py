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


class Individual(Agent):
    def __init__(self, idx, radius, budget, group=None, group_num=2, max_accel, dist_action=False):
        '''
        idx: index of the agent
        radius: radius of the agent 
        group: group of the agent (recruiters or freelancers)
        group_num: number of agents in the group (recruiters or freelancers)

        # we may not need these
        max_accel: maximum acceleration of the agent
        dist_action: whether the action is discrete or continuous
        '''
        self._idx = idx
        self._radius = radius
        self._budget = budget # budget of the agent (only recruiters have budget) set to None for freelancers
        
        self._max_accel = max_accel
        self.dist_action = dist_action

        # agent's position
        self._position = None

        # agent's velocity (We don't need this)
        # self._velocity = None

        # claim the agent to be in a group (recruiters or freelancers)
        self._group = group

        # set the agent to be visible
        self._visible = True
        
    @property
    def budget(self):
        if self._budget is None and self._group == 'recruiters':
            raise ValueError("Budget is not set")
        else:
            return self._budget

    @property
    def observation_space(self):
    # TODO: define the observation space for the agent
        pass
    
    @property
    def action_space(self):
    # TODO: define the action space for the agent
        pass

    @property
    def position(self):
        assert self._position is not None
        return self._position
    
    @property
    def visible(self):
        return self._visible

    # hide the agent if it is needed
    def hide(self):
        self._visible = False
    
    # reveal the agent if it is needed
    def reveal(self):
        self._visible = True

    # set the position of the agent
    def set_position(self, pos):
        assert pos.shape == (2,)
        self._position = pos


class Jobmatching():
    """A Bipartite Networked Multi-agent Environment."""

    def __init__(self, n_agents_recruiters = 2, n_agents_freelancers = 2, radius = 0.015, budget, match_reward, 
                 use_groudtruth=False, local_ratio, recruiters_max_accel, freelancers_max_accel, step_count, window_size=2, **kwargs):
        '''
        n_agents_recruiters: Number of agents in group A (recruiters)
        n_agents_freelancers: Number of agents in group B (freelancers)
        radius: agent base radius. recruiters: radius, freelancers: radius
        match_reward: reward for matching
        use_groudtruth: whether to use groundtruth for matching
        local_ratio: Proportion of reward allocated locally vs distributed globally among all agents (We may not need this)
        recruiters_max_accel: recruiters maximum acceleration (maximum action size)
        max_cycles: After max_cycles steps all agents will return done
        '''
        self.n_agents_recruiters = n_agents_recruiters
        self.n_agents_freelancers = n_agents_freelancers
        self.radius = radius
        self.budget = budget # need to specify the exact budget for recruiters
        self.match_reward = match_reward # need to specify the exact reward for matching
        self.use_groudtruth = use_groudtruth
        self.recruiters_max_accel = recruiters_max_accel
        self.freelancers_max_accel = freelancers_max_accel
        self.seed()
        self._recruiters = [
            Individual(recruiters_idx + 1, self.radius, budget = self.budget[recruiters_idx], group='recruiters',
                   group_num=self.n_agents_recruiters, max_accel = self.recruiters_max_accel , dist_action=self.dist_action)
            for recruiters_idx in range(self.n_agents_recruiters)
        ]

        self._freelancers = [
            Individual(freelancers_idx + 1, self.radius, budget = None, group='freelancers',
                   group_num=self.n_agents_freelancers, max_accel = self.freelancers_max_accel , dist_action=self.dist_action)
            for freelancers_idx in range(self.n_agents_freelancers)
        ]
       
        # TODO: specify the colors for recruiters and freelancers 
        # I am not sure if this is the right way to do it
        # just copied from the original code 
        self.colors = [
            (192, 64, 64),
            (64, 192, 64),
            (64, 64, 192),
            (192, 192, 64),
            (192, 64, 192),
            (64, 192, 192),
        ]

        self.action_space_recruiters = [agent.action_space for agent in self._recruiters]
        self.action_space_freelancers = [agent.action_space for agent in self._freelancers]
        self.observation_space_recruiters = [
            agent.observation_space for agent in self._recruiters]
        self.observation_space_freelancers = [
            agent.observation_space for agent in self._freelancers]
        print("observation space recruiters", self.observation_space_recruiters)
        print("observation space freelancers", self.observation_space_freelancers)

        # some parameters to render the window?
        self.renderOn = False
        self.pixel_scale = 30 * 25
        self.window_size = window_size

        self.cycle_time = 1.0
        self.frames = 0
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

    # generate coordinates for recruiters and freelancers
    # TODO: need to make sure that the coordinates are not overlapping with each other 
    # TODO: make sure recruiters are on the left side and freelancers are on the right side
    def _generate_coord(self, radius):
        coord = self.np_random.rand(2)
        # # Create random coordinate that avoids obstacles
        # while ssd.cdist(coord[None, :], self.obstacle_coords) <= radius * 2 + self.obstacle_radius:
        #     coord = self.np_random.rand(2)
        return coord
    
    # TODO: Pending editing
    # remove the obstacles from the original code because we don't need them
    def reset(self):
        self.frames = 0

        # Initialize recruiters and freelancers
        for recruiter in self._recruiters:
            recruiter.set_position(self._generate_coord(recruiter._radius))

        for freelancer in self._freelancers:
            freelancer.set_position(self._generate_coord(freelancer._radius))
        
        # initialize the rewards for recruiters and freelancers
        rewards = np.zeros(self.n_agents_recruiters + self.n_agents_freelancers)

        # TODO: adapt the following four lines from the original code
        sensor_features, collided_pursuer_evader, collided_pursuer_poison, rewards \
            = self.collision_handling_subroutine(rewards, True)
        obs_list = self.observe_list(
            sensor_features, collided_pursuer_evader, collided_pursuer_poison)

        
        self.last_rewards = [np.float64(0) for _ in range(self.n_agents_recruiters + self.n_agents_freelancers)]
        self.control_rewards = [0 for _ in range(self.n_agents_recruiters + self.n_agents_freelancers)]
        self.last_dones = [False for _ in range(self.n_agents_recruiters + self.n_agents_freelancers)]
        self.last_obs = obs_list
        return obs_list[0]

    # TODO: Pending editing
    def observe_list(self, sensor_feature, is_colliding_evader, is_colliding_poison):
        obslist = []
        for pursuer_idx in range(self.n_pursuers):
            one_hot = np.zeros(self.n_pursuers)
            one_hot[pursuer_idx] = 1.0
            position = self._pursuers[pursuer_idx].position
            obslist.append(
                np.concatenate([
                    sensor_feature[pursuer_idx, ...].ravel(), [
                        float((is_colliding_evader[pursuer_idx, :]).sum() > 0), float((
                            is_colliding_poison[pursuer_idx, :]).sum() > 0)
                    ], one_hot, position
                ]))
        return obslist  

    # TODO: Pending editing
    def convert_action(self, action):
        '''convert a discrete action to continuous acceleration'''
        action_map = np.array([[0,0], [1, 0], [0, 1], [-1, 0], [0, -1],
                            [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5],[-0.5, -0.5]])
        return action_map[action] * 0.05 # *self.pursuer_max_accel*0.5

    # TODO: Pending editing
    def step(self, action, agent_id, is_last):
        if self.dist_action:
            action = self.convert_action(action)
        else:
            action = np.asarray(action)
            action = action.reshape(2)
            speed = np.linalg.norm(action)
            if speed > self.pursuer_max_accel:
                # Limit added thrust to self.pursuer_max_accel
                action = action / speed * self.pursuer_max_accel

        p = self._pursuers[agent_id]
        if self.dist_action:
            p.set_velocity(action) # if the action is the velocity instead of acceleration
        else:
            p.set_velocity(p.velocity + action)
        
        p.set_position(p.position + self.cycle_time * p.velocity)

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * math.sqrt((action ** 2).sum())
        # Average thrust penalty among all agents, and assign each agent global portion designated by (1 - local_ratio)
        self.control_rewards = (accel_penalty / self.n_pursuers) * np.ones(self.n_pursuers) * (1 - self.local_ratio)
        # Assign the current agent the local portion designated by local_ratio
        self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        if is_last:
            def move_objects(objects):
                for obj in objects:
                    # Move objects
                    obj.set_position(obj.position + self.cycle_time * obj.velocity)
                    # Bounce object if it hits a wall
                    for i in range(len(obj.position)):
                        if obj.position[i] >= 1 or obj.position[i] <= 0:
                            obj.position[i] = np.clip(obj.position[i], 0, 1)
                            obj.velocity[i] = -1 * obj.velocity[i]

            move_objects(self._evaders)
            move_objects(self._poisons)

            rewards = np.zeros(self.n_pursuers)
            sensorfeatures, collisions_pursuer_evader, collisions_pursuer_poison, rewards = self.collision_handling_subroutine(rewards, is_last)
            obs_list = self.observe_list(
                sensorfeatures, collisions_pursuer_evader, collisions_pursuer_poison)
            self.last_obs = obs_list

            # new reward: negative if there are foods that have not been eaten
            for g in range(self.n_groups):
                visibles = [e.visible for e in self._evaders[g*self.n_evaders_pergroup:(g+1)*self.n_evaders_pergroup]]
                rewards[g] -= np.array(visibles).sum() * self.food_alive_penalty

            local_reward = rewards
            global_reward = local_reward.mean()
            # Distribute local and global rewards according to local_ratio
            self.last_rewards = local_reward * self.local_ratio + global_reward * (1 - self.local_ratio)

            self.frames += 1

        if self.comm:
            return self.observe(agent_id), self.communicate(agent_id)
        else:
            return self.observe(agent_id)

    def observe(self, agent):
        return np.array(self.last_obs[agent], dtype=np.float32)

    def communicate(self, agent):
        return np.array(self.last_comm[agent], dtype=np.float32)    


    # The following functions are for rendering the environment in pygame
    def draw_background(self):
        # -1 is building pixel flag
        color = (255, 255, 255)
        rect = pygame.Rect(0, 0, self.window_size * self.pixel_scale, self.window_size * self.pixel_scale)
        pygame.draw.rect(self.screen, color, rect)

    def draw_recruiters(self):
        for recruiter in self._recruiters:
            x, y = recruiter.position
            center = (int(self.window_size * self.pixel_scale * x),
                      int(self.window_size * self.pixel_scale * y))
            color = self.colors[recruiter._group%len(self.colors)]
            pygame.draw.circle(self.screen, color, center, self.pixel_scale * self.radius)

    def draw_freelancers(self):
        for freelancer in self._freelancers:
            x, y = freelancer.position
            center = (int(self.window_size * self.pixel_scale * x),
                      int(self.window_size * self.pixel_scale * y))
            color = self.colors[freelancer._group%len(self.colors)]
            pygame.draw.circle(self.screen, color, center, self.pixel_scale * self.radius)

    def render(self, mode="human"):
        if not self.renderOn:
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.window_size*self.pixel_scale, self.window_size*self.pixel_scale))
            else:
                self.screen = pygame.Surface((self.window_size*self.pixel_scale, self.window_size*self.pixel_scale))
            self.renderOn = True

        self.draw_background()
        self.draw_recruiters()
        self.draw_freelancers()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if mode == "human":
            pygame.display.flip()
        return np.transpose(new_observation, axes=(1, 0, 2)) if mode == "rgb_array" else None
