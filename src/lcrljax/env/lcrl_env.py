from typing import Any, Dict, List, Tuple
from bsuite.environments.base import Environment

from lcrljax.automata.ldba import LDBA, JaxLDBA
import dm_env
from dm_env import TimeStep


class LCRLEnv(Environment):
    """
    This is an environment which implements some of the constructions
    we need for the LCRL algorithm.

    This environment can wrap another environment, and will hide much of the
    complexity of the LCRL algorithm.

    The underlying environment doesn't have an understanding of the
    automata states so we need to wrap it.

    We have 2 subsets of actions, one for the base environment and one for the epsilon transitions.

    The base environment has a fixed set of actions, and the epsilon actions contain one for each state
    which has an epsilon transition to it.

    If this action is chosen, the environment will take the epsilon transition if it is valid.
    """

    def __init__(self, automata: JaxLDBA, base_env: Environment):
        self.automata = automata
        self.frontier = automata.initial_frontier
        self.base_env = base_env
        self.automata_state = automata.initial_state()
        self.previous_inner_state = self.base_env.reset()
        self.epsilon_transitions: Dict[int, int] = {}

    def convert_reward(self, timestep: dm_env.TimeStep) -> TimeStep:
        return dm_env.TimeStep(
            timestep.step_type,
            timestep.observation,
            self._reward(timestep),
            timestep.discount,
        )

    def combine_automata_state(self, timestep: dm_env.TimeStep) -> TimeStep:
        return dm_env.TimeStep(
            timestep.step_type,
            (timestep.observation, self.automata_state),
            timestep.reward,
            timestep.discount,
        )

    def _reward(self, timestep: dm_env.TimeStep) -> float:
        """Returns a reward for the given `timestep`."""
        q_prime = timestep.observation[1]
        reward = 1 if self.frontier[q_prime] else 0
        self.frontier = self.automata.accepting_frontier_function(q_prime, self.frontier)
        return reward

    def _reset(self) -> TimeStep:
        """Returns a `timestep` namedtuple as per the regular `reset()` method."""
        self.frontier = self.automata.initial_frontier
        self.automata_state = self.automata.initial_state()
        self.previous_inner_state = self.base_env.reset()
        return self.convert_reward(self.combine_automata_state(self.previous_inner_state))

    def _step(self, action: int) -> TimeStep:
        """Returns a `timestep` namedtuple as per the regular `step()` method."""
        if action in self.epsilon_transitions:
            self.automata_state = self.automata.step(self.automata_state, self.epsilon_transitions[action])
            return self.convert_reward(self.combine_automata_state(self.previous_inner_state))
        return self.convert_reward(self.combine_automata_state(self.base_env.step(action)))

    def bsuite_info(self) -> Dict[str, Any]:
        return {}

    def observation_spec(self):
        return super().observation_spec()

    def action_spec(self):
        return super().action_spec()
