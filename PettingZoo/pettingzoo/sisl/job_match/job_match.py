# not used

'''
from .job_match_base import Jobmatching_step1 as _env_1
from .job_match_base import Jobmatching_step2 as _env_2
from .job_match_base import Jobmatching_step3 as _env_3

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn


def env(**kwargs):
    env = raw_env(**kwargs)
    if not kwargs["dist_action"]:
        env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):

    # metadata = {'render.modes': ['human', "rgb_array"], 'name': 'job_match_v0'}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = _env(*args, **kwargs)
        self.recruiters = ["recruiter_" + str(r) for r in range(self.env.n_agents_recruiters)]
        self.freelancers = ["freelancer_" + str(c) for c in range(self.env.n_agents_freelancers)]
        self.recruiters_name_mapping = dict(zip(self.recruiters, list(range(self.env.n_agents_recruiters))))
        self.freelancers_name_mapping = dict(zip(self.freelancers, list(range(self.env.n_agents_freelancers))))
        self.recruiter_selector = agent_selector(self.recruiters)
        self.freelancer_selector = agent_selector(self.freelancers)
        # spaces
        self.action_space_1 = dict(zip(self.freelancers, self.env.action_space_1))
        self.action_space_2 = dict(zip(self.recruiters, self.env.action_space_2))
        self.action_space_3 = dict(zip(self.freelancers, self.env.action_space_3))
        self.observation_space_1 = dict(zip(self.freelancers, self.env.observation_space_1))
        self.observation_space_2 = dict(zip(self.recruiters, self.env.observation_space_2))
        self.observation_space_3 = dict(zip(self.freelancers, self.env.observation_space_3))
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

        for r in self.rewards:
            self.rewards[r] = self.env.control_rewards[self.agent_name_mapping[r]]
        if is_last:
            for r in self.rewards:
                self.rewards[r] += self.env.last_rewards[self.agent_name_mapping[r]]

        if self.env.frames >= self.env.max_cycles:
            self.dones = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self.dones = dict(zip(self.agents, self.env.last_dones))
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])
'''