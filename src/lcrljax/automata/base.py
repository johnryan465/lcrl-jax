from abc import ABC, abstractmethod
from typing import Generic, Optional, Set, TypeVar


State = TypeVar("State")
Alphabet = TypeVar("Alphabet")
SetState = TypeVar("SetState")
SetAlphabet = TypeVar("SetAlphabet")


class Automata(ABC, Generic[State, Alphabet, SetState, SetAlphabet]):
    @abstractmethod
    def alphabet(self) -> SetAlphabet:
        """
        Returns the alphabet of the automata.
        """
        pass

    @abstractmethod
    def states(self) -> SetState:
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
