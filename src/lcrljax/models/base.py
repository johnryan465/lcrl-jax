from abc import ABC, abstractmethod
import collections
from typing import Tuple


Params = collections.namedtuple("Params", "online target")
ActorState = collections.namedtuple("ActorState", "count")
ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
LearnerState = collections.namedtuple("LearnerState", "count opt_state")
Data = collections.namedtuple("Data", "obs_tm1 a_tm1 r_t discount_t obs_t")


class RLModel(ABC):
    @abstractmethod
    def actor_step(self, params: Params, env_output, actor_state: ActorState, key, evaluation) -> Tuple[ActorOutput, ActorState]:
        pass

    @abstractmethod
    def learner_step(
            self, params: Params, data: Data, learner_state: LearnerState, unused_key: int) -> Tuple[
            Params, LearnerState]:
        pass

    @abstractmethod
    def initial_params(self, key: int) -> Params:
        pass

    @abstractmethod
    def initial_actor_state(self) -> ActorState:
        pass

    @abstractmethod
    def initial_learner_state(self, params: Params) -> LearnerState:
        pass
