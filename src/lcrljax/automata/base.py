from abc import ABC, abstractmethod
from typing import Generic, Optional, Set, TypeVar


State = TypeVar("State")
Alphabet = TypeVar("Alphabet")


class Automata(ABC, Generic[State, Alphabet]):
    @abstractmethod
    def alphabet(self) -> Set[Alphabet]:
        """
        Returns the alphabet of the automata.
        """
        pass

    @abstractmethod
    def states(self) -> Set[State]:
        """
        Returns the states of the automata.
        """
        pass

    @abstractmethod
    def initial_state(self) -> State:
        """
        Returns the initial state of the automata.
        """
        pass

    @abstractmethod
    def next_states(self, state: State, symbol: Optional[Alphabet]) -> Set[State]:
        """
        Returns the next states of the automata.
        """
        pass

    @abstractmethod
    def accept_run(self, run: Set[Alphabet]) -> bool:
        """
        Returns true if the run is accepted by the automata.
        """
        pass
