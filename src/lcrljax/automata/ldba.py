from lcrljax.automata.base import Automata

from typing import Callable, Optional, Set, TypeVar


State = TypeVar("State")
Alphabet = TypeVar("Alphabet")


class GBA(Automata[State, Alphabet]):
    """
    Generalized BA (Generalized Buchi Automata)
    Omega: a finite set of states
    q_0: a state in Omega (initial state)
    Sigma: 2^n symbols (alphabet)
    Delta: a transition relation on Omega x Sigma to 2^Omega
    F: is the set of accepting conditions

    The automata accepts a run when the following is true:
    all of the accepting conditions are satisfied by at least one
    of the states which occur infinitely often in the run.
    """

    def __init__(
            self, omega: Set[State],
            q_0: State, sigma: Set[Alphabet],
            delta: Callable[[State, Optional[Alphabet]],
                            Set[State]]):
        self.q_0 = q_0
        self.omega = omega
        self.sigma = sigma
        self.delta = delta

    def alphabet(self) -> Set[Alphabet]:
        return self.sigma

    def states(self) -> Set[State]:
        return self.omega

    def initial_state(self) -> State:
        return self.q_0

    def next_states(self, state: State, symbol: Optional[Alphabet]) -> Set[State]:
        return self.delta(state, symbol)

    def accept_run(self, run: Set[Alphabet]) -> bool:
        return False


class LDBA(Automata[State, Alphabet]):
    """
    Linear Deterministic Buchi Automata

    This is a GBA in which the states can be partitioned into 2 disjoint sets.
    Q_D and Q_N

    Q_D (Accepting States)
    Q_N (Initial States)

    - All states in Q_D |Delta(q,a)| = 1 and Delta(q,a) in Q_D
    - F_i subset of Q_D
    - q_0 in Q_N, all transitions from Q_N to Q_D are epislon transitions
    """
    pass
