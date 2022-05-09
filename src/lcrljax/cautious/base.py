from abc import ABC, abstractmethod
import jax.numpy as jnp


class ActionModification(ABC):
    """
    Base class for modification of action probabilies while training.

    """
    @abstractmethod
    def modify(self, state: jnp.ndarray, action_probs: jnp.ndarray) -> jnp.ndarray:
        """
        Modifies the action probabilities.
        """
        pass


class NoActionModification(ActionModification):
    """
    No action modification.
    """

    def modify(self, state: jnp.ndarray, action_probs: jnp.ndarray) -> jnp.ndarray:
        return action_probs


class DiscreteSafeExploration(ActionModification):
    """
    This class implements the modification procedure described in "Cautious Learning with Safe Exploration"
    """
    pass


class ContinuousSafeExploration(ActionModification):
    """
    This class implements a new proposed modification procedure for continous safe exploration."""
