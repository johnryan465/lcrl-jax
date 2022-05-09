from dataclasses import dataclass
from typing import Callable, Generic, List, TypeVar

State = TypeVar("State")
Action = TypeVar("Action")


@dataclass
class MDP(Generic[State, Action]):
    """
    A Markov Decision Process (MDP)

    """
    states: List[State]
    actions: List[Action]
    transitions: Callable[[State, Action, State], float]
    rewards: Callable[[State, Action, State], float]
