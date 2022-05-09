from abc import abstractmethod
from lcrljax.automata.base import Automata
import jumpy as jp
from typing import Callable, List, Optional, Set, TypeVar
import jax

State = TypeVar("State")
Alphabet = TypeVar("Alphabet")
SetState = TypeVar("SetState")
SetAlphabet = TypeVar("SetAlphabet")


class BaseLDBA(Automata[State, Alphabet, SetState, SetAlphabet]):
    """
    Linear Deterministic Buchi Automata

    This is a GBA in which the states can be partitioned into 2 disjoint sets.
    Q_D and Q_N

    Q_D(Accepting States)
    Q_N(Initial States)

    - All states in Q_D | Delta(q, a) | = 1 and Delta(q, a) in Q_D
    - F_i subset of Q_D
    - q_0 in Q_N, all transitions from Q_N to Q_D are epislon transitions
    """
    @abstractmethod
    def accepting_frontier_function(self, q: State, F: SetState) -> SetState:
        """
        Returns the accepting frontier function of the automata."""

    @abstractmethod
    def step(self, q: State, a: Alphabet) -> State:
        """
        Returns the next state of the automata."""


class LDBA(BaseLDBA[State, Alphabet, Set[State], Set[Alphabet]]):

    def __init__(
            self, omega: Set[State],
            q_0: State, sigma: Set[Alphabet],
            delta: Callable[[State, Optional[Alphabet]],
                            Set[State]]):
        self.q_0 = q_0
        self.omega = omega
        self.sigma = sigma
        self.delta = delta
        self.fs: List[Set[State]] = []

    def accepting_frontier_function(self, q: State, F: Set[State]) -> Set[State]:
        for i in range(len(self.fs)):
            if q in self.fs[i]:
                if F != self.fs[i]:
                    return F.difference(self.fs[i])
                else:
                    return frozenset().union(*self.fs).difference(self.fs[i])  # type: ignore

        return F

    def alphabet(self) -> Set[Alphabet]:
        return self.sigma

    def states(self) -> Set[State]:
        return self.omega

    def initial_state(self) -> State:
        return self.q_0


class JaxLDBA(BaseLDBA[int, int, jp.ndarray, jp.ndarray]):
    def __init__(self, num_states: int, num_actions: int, conditions: jp.ndarray):
        self.num_states = num_states
        self.num_actions = num_actions
        self.fs = conditions
        self.union_fs: jp.ndarray = jax.numpy.array([])

    def accepting_frontier_function(self, q: int, F: jp.ndarray) -> jp.ndarray:
        idx = jax.numpy.nonzero(self.fs[:, q])[0]
        equals_conditions = jax.vmap(jax.numpy.equal, in_axes=(None, 0))(q, self.fs)  # type: ignore
        if len(idx) > 0:
            i = idx[0]
            if equals_conditions[q]:
                return jax.numpy.logical_and(F, jax.numpy.logical_not(self.fs[i]))
            else:
                return jax.numpy.logical_and(self.union_fs, jax.numpy.logical_not(self.fs[i]))
        return F

    def states(self) -> jp.ndarray:
        return jax.numpy.arange(self.num_states)

    def alphabet(self) -> jp.ndarray:
        return jax.numpy.arange(self.num_actions)

    def initial_state(self) -> int:
        return 0

    def step(self, q: int, a: int) -> int:
        return q

    @property
    def initial_frontier(self) -> jp.ndarray:
        return self.union_fs
