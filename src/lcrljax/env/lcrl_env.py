from functools import partial
from typing import Any, Dict, List, Tuple
from bsuite.environments.base import Environment

from lcrljax.automata.ldba import LDBA, JaxLDBA
import dm_env
from dm_env import TimeStep
import jax
from brax.envs import env
import jumpy as jp
from dm_env import specs
import brax


class LCRLEnv(env.Env):
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

    def __init__(self, automata: JaxLDBA, base_env: env.Env):
        self.automata = automata
        self.frontier = automata.initial_frontier
        self.base_env = base_env
        self.automata_state = automata.initial_state()
        self.num_actions = self.base_env.action_size
        self.epsilon_transitions: Dict[int, int] = {}

    @partial(jax.jit, static_argnums=(0,))
    def modify_reward(self, state: env.State) -> env.State:
        q_prime = state.info["automata_state"]
        frontier = state.info["frontier"]
        # print(frontier)
        # print(q_prime)
        reward = frontier[q_prime].astype(jp.float32)
        frontier = self.automata.accepting_frontier_function(q_prime, frontier)
        return state.replace(reward=reward, info=dict(state.info, **{"frontier": frontier}))  # type: ignore

    @partial(jax.jit, static_argnums=(0,))
    def add_automata_state(self, state: env.State, automata_state: jp.ndarray) -> env.State:
        return state.replace(info=dict(state.info, **{"automata_state": automata_state}))  # type: ignore

    @partial(jax.jit, static_argnums=(0,))
    def add_frontier(self, state: env.State, frontier: jp.ndarray) -> env.State:
        return state.replace(info=dict(state.info, **{"frontier": frontier}))  # type: ignore

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jp.ndarray) -> env.State:
        """Returns a `timestep` namedtuple as per the regular `reset()` method."""
        frontier = self.automata.initial_frontier
        automata_state = self.automata.initial_state()
        state = self.base_env.reset(rng)

        return self.modify_reward(self.add_frontier(self.add_automata_state(state, automata_state), frontier))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Returns a `timestep` namedtuple as per the regular `step()` method."""
        automata_state = state.info["automata_state"]
        next_automata_state = self.automata.step(automata_state, action - self.num_actions)

        def true_fn(x):
            return x

        def false_fn(x):
            print(x, action)
            return self.base_env.step(x, action)
        print(action)
        print(self.num_actions)
        print(jax.numpy.greater_equal(action, self.num_actions).shape)
        state = jax.lax.cond(
            jax.numpy.greater_equal(action, self.num_actions),
            true_fn, false_fn, state)
        return self.modify_reward(self.add_automata_state(state, next_automata_state))

    @ property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        return self.base_env.observation_size

    @ property
    def action_size(self) -> int:
        return self.base_env.action_size + len(self.epsilon_transitions)

    @ property
    def sys(self) -> brax.System:
        return self.base_env.sys

    def action_spec(self) -> specs.Array:
        return specs.Array(shape=(self.action_size,), dtype=jp.float32)

    def observation_spec(self) -> specs.Array:
        return specs.Array(shape=(self.observation_size,), dtype=jp.float32)
