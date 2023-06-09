from .job_match.job_match_base import Jobmatching_step1 as _env_1
from .job_match.job_match_base import Jobmatching_step2 as _env_2
from .job_match.job_match_base import Jobmatching_step3 as _env_3
from .job_match.job_match_base import ToyExample as _env_toy

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
import numpy as np
from gym import spaces
from collections import defaultdict
import copy
from pettingzoo.utils.env import ParallelEnv
# from ..utils.conversions import to_parallel_wrapper


# def env(**kwargs):
#     env = raw_env(**kwargs)
#     if not kwargs["dist_action"]:
#         env = wrappers.ClipOutOfBoundsWrapper(env)
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env

def env_1(**kwargs):
    env = raw_env_1(**kwargs)
    return env

def env_2(**kwargs):
    env = raw_env_2(**kwargs)
    return env

def env_3(**kwargs):
    env = raw_env_3(**kwargs)
    return env

def env_toy(**kwargs):
    env = raw_env_toy(**kwargs)
    return env

class to_parallel_wrapper_comm(ParallelEnv):
    def __init__(self, aec_env):
        self.aec_env = aec_env
        self.observation_spaces = aec_env.observation_spaces
        self.action_spaces = aec_env.action_spaces
        # self.communication_spaces = aec_env.communication_spaces
        self.possible_agents = aec_env.possible_agents
        self.metadata = aec_env.metadata
        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = self.aec_env.state_space
        except AttributeError:
            pass

    @property
    def unwrapped(self):
        return self.aec_env.unwrapped

    def seed(self, seed=None):
        return self.aec_env.seed(seed)

    def reset(self):
        self.aec_env.reset()
        self.agents = self.aec_env.agents
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents if not self.aec_env.dones[agent]}
        # communications = {agent: self.aec_env.communicate(agent) for agent in self.aec_env.agents}
        return observations # , communications

    def step(self, actions):
        while self.aec_env.agents and self.aec_env.dones[self.aec_env.agent_selection]:
            self.aec_env.step(None)

        rewards = {a: 0 for a in self.aec_env.agents}
        dones = {}
        infos = {}
        observations = {}

        for agent in self.aec_env.agents:
            assert agent == self.aec_env.agent_selection, f"expected agent {agent} got agent {self.aec_env.agent_selection}, agent order is nontrivial"
            # obs, rew, done, info = self.aec_env.last()
            self.aec_env.step(actions[agent])
        for agent in self.aec_env.agents:
            rewards[agent] = self.aec_env.rewards[agent]
        new_actions = {agent: None for agent in self.aec_env.agents}
        for agent in self.aec_env.agents:
            new_actions[agent] = self.aec_env.new_actions[agent]

        dones = dict(**self.aec_env.dones)
        infos = dict(**self.aec_env.infos)
        self.agents = self.aec_env.agents
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents}
        # communications = {agent: self.aec_env.communicate(agent) for agent in self.aec_env.agents}
        # print("rewards: {}".format(rewards))
        # for env1 and env2, should get rewards from env3
        return observations, rewards, dones, infos, new_actions # observations, communications, rewards, dones, infos

    # def render(self, mode="human"):
    #     return self.aec_env.render(mode)

    def state(self):
        return self.aec_env.state()

    def close(self):
        return self.aec_env.close()

class to_parallel_wrapper_comm_toy(ParallelEnv):
    def __init__(self, aec_env):
        self.aec_env = aec_env
        self.observation_spaces = aec_env.observation_spaces
        self.action_spaces = aec_env.action_spaces
        self.possible_agents = aec_env.possible_agents
        self.metadata = aec_env.metadata
        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = self.aec_env.state_space
        except AttributeError:
            pass

    @property
    def unwrapped(self):
        return self.aec_env.unwrapped
    
    def reset(self):
        self.aec_env.reset()
        self.agents = self.aec_env.agents
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents if not self.aec_env.dones[agent]}
        return observations

    def step(self, actions):
        while self.aec_env.agents and self.aec_env.dones[self.aec_env.agent_selection]:
            self.aec_env.step(None)

        rewards = {a: 0 for a in self.aec_env.agents}
        dones = {}
        infos = {}
        observations = {}

        for agent in self.aec_env.agents:
            assert agent == self.aec_env.agent_selection, f"expected agent {agent} got agent {self.aec_env.agent_selection}, agent order is nontrivial"
            # obs, rew, done, info = self.aec_env.last()
            self.aec_env.step(actions[agent])
        for agent in self.aec_env.agents:
            rewards[agent] = self.aec_env.rewards[agent]
    
        dones = dict(**self.aec_env.dones)
        infos = dict(**self.aec_env.infos)
        self.agents = self.aec_env.agents
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents}
        # communications = {agent: self.aec_env.communicate(agent) for agent in self.aec_env.agents}
        # print("rewards: {}".format(rewards))
        # for env1 and env2, should get rewards from env3
        return observations, rewards, dones, infos# observations, communications, rewards, dones, infos

    def state(self):
        return self.aec_env.state()

    def close(self):
        return self.aec_env.close()

def parallel_comm_wrapper_fn(env_fn):
    def par_fn(**kwargs):
        print(kwargs)
        env = env_fn(**kwargs)
        env = to_parallel_wrapper_comm(env)
        return env
    return par_fn

def parallel_comm_wrapper_fn_toy(env_fn):
    def par_fn(**kwargs):
        print(kwargs)
        env = env_fn(**kwargs)
        env = to_parallel_wrapper_comm_toy(env)
        return env
    return par_fn

parallel_env_1 = parallel_comm_wrapper_fn(env_1)
parallel_env_2 = parallel_comm_wrapper_fn(env_2)
parallel_env_3 = parallel_comm_wrapper_fn(env_3)
parallel_env_toy = parallel_comm_wrapper_fn_toy(env_toy)

class raw_env_1(AECEnv):

    metadata = {}

    # metadata = {'render.modes': ['human', "rgb_array"], 'name': 'foodcollector_v1'}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = _env_1(*args, **kwargs)
        self.agents = ["freelancer_" + str(r) for r in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(
            zip(self.agents, self.env.observation_space))
        self.new_actions = dict(zip(self.agents, [None for _ in self.agents]))
        self.has_reset = False

    def seed(self, seed=None):
        self.env.seed(seed)

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    def reset(self):
        self.has_reset = True
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)
        self.new_actions[agent] = self.env.new_actions[self.agent_name_mapping[agent]]
        # for r in self.rewards:
        #     self.rewards[r] = self.env.control_rewards[self.agent_name_mapping[r]]
        # if is_last:
        #     for r in self.rewards:
        #         self.rewards[r] += self.env.last_rewards[self.agent_name_mapping[r]]

        if self.env.step_count >= self.env.max_cycles:
            self.dones = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self.dones = dict(zip(self.agents, self.env.last_dones))
        # self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        # self._accumulate_rewards()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])


class raw_env_2(AECEnv):

    metadata = {}
    # metadata = {'render.modes': ['human', "rgb_array"], 'name': 'foodcollector_v1'}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = _env_2(*args, **kwargs)
        self.agents = ["recruiter_" + str(r) for r in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(
            zip(self.agents, self.env.observation_space))
        self.new_actions = dict(zip(self.agents, [None for _ in self.agents]))
        self.has_reset = False

    def seed(self, seed=None):
        self.env.seed(seed)

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    def reset(self):
        self.has_reset = True
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)
        self.new_actions[agent] = self.env.new_actions[self.agent_name_mapping[agent]]
        # for r in self.rewards:
        #     self.rewards[r] = self.env.control_rewards[self.agent_name_mapping[r]]
        if is_last:
            for r in self.rewards:
                self.rewards[r] += self.env.last_rewards[self.agent_name_mapping[r]]

        if self.env.step_count >= self.env.max_cycles:
            self.dones = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self.dones = dict(zip(self.agents, self.env.last_dones))
        # self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        # self._accumulate_rewards()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])


class raw_env_3(AECEnv):

    metadata = {}
    # metadata = {'render.modes': ['human', "rgb_array"], 'name': 'foodcollector_v1'}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = _env_3(*args, **kwargs)
        self.agents = ["freelancer_" + str(r) for r in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(
            zip(self.agents, self.env.observation_space))
        self.new_actions = dict(zip(self.agents, [None for _ in self.agents]))
        self.has_reset = False

    def seed(self, seed=None):
        self.env.seed(seed)

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    def reset(self):
        self.has_reset = True
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)
        self.new_actions[agent] = self.env.new_actions[self.agent_name_mapping[agent]]
        # for r in self.rewards:
        #     self.rewards[r] = self.env.control_rewards[self.agent_name_mapping[r]]
        if is_last:
            for r in self.rewards:
                self.rewards[r] = self.env.last_rewards[self.agent_name_mapping[r]]

        if self.env.step_count >= self.env.max_cycles:
            self.dones = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self.dones = dict(zip(self.agents, self.env.last_dones))
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])


class raw_env_toy(AECEnv):

    metadata = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = _env_toy(*args, **kwargs)
        self.agents = ["agent_" + str(r) for r in range(self.env.n_agents_freelancers)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.env.n_agents_freelancers))))
        self._agent_selector = agent_selector(self.agents)
        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(
            zip(self.agents, self.env.observation_space))
        # self.new_actions = dict(zip(self.agents, [None for _ in self.agents]))
        self.has_reset = False

    def reset(self):
        self.has_reset = True
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))\

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)
        # self.new_actions[agent] = self.env.new_actions[self.agent_name_mapping[agent]]
        # for r in self.rewards:
        #     self.rewards[r] = self.env.control_rewards[self.agent_name_mapping[r]]
        if is_last:
            for r in self.rewards:
                self.rewards[r] = self.env.last_rewards[self.agent_name_mapping[r]]

        self.dones = dict(zip(self.agents, self.env.last_dones))
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])