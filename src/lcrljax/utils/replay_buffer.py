import collections
import numpy as np
import random
from absl import flags

from lcrljax.models.base import Data


FLAGS = flags.FLAGS
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")


class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, env_output, action):
        self._prev = self._latest
        self._action = action
        self._latest = env_output

        if action is not None:
            self.buffer.append(
                (self._prev.obs, self._action, self._latest.reward,
                 1., self._latest.obs))

    def sample(self, batch_size) -> Data:
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
            *random.sample(self.buffer, batch_size))
        return Data(np.stack(obs_tm1), np.asarray(a_tm1), np.asarray(r_t),
                    np.asarray(discount_t) * FLAGS.discount_factor, np.stack(obs_t))

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)
