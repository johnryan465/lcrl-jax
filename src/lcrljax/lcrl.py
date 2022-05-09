"""
To do LCRL we need to do the following:

- Create a product state of the MDP and the automata.
- Create the modified reward function.


"""


from typing import Tuple
import jumpy as jp
import jax
from lcrljax.automata.ldba import JaxLDBA
from lcrljax.env.lcrl_env import LCRLEnv
from lcrljax.models.base import RLModel
from bsuite.environments.base import Environment

from lcrljax.utils.replay_buffer import ReplayBuffer

import jax

from absl import app
from absl import flags
from bsuite.environments import catch
from bsuite.environments.base import Environment
import haiku as hk


from lcrljax.models.dqn import DQN
from lcrljax.utils.replay_buffer import ReplayBuffer


FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("train_episodes", 301, "Number of train episodes.")
flags.DEFINE_integer("batch_size", 32, "Size of the training batch")
flags.DEFINE_float("target_period", 50, "How often to update the target net.")
flags.DEFINE_integer("replay_capacity", 2000, "Capacity of the replay buffer.")
flags.DEFINE_integer("hidden_units", 50, "Number of network hidden units.")
flags.DEFINE_float("epsilon_begin", 1., "Initial epsilon-greedy exploration.")
flags.DEFINE_float("epsilon_end", 0.01, "Final epsilon-greedy exploration.")
flags.DEFINE_integer("epsilon_steps", 1000, "Steps over which to anneal eps.")
flags.DEFINE_float("learning_rate", 0.005, "Optimizer learning rate.")
flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 50,
                     "Number of episodes between evaluations.")


def run_loop(
        agent: RLModel,
        environment: LCRLEnv,
        accumulator: ReplayBuffer,
        seed: int,
        batch_size: int,
        train_episodes: int,
        evaluate_every: int,
        eval_episodes: int):
    """A simple run loop for examples of reinforcement learning with rlax."""

    # Init agent.
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    params = agent.initial_params(next(rng))  # type: ignore
    learner_state = agent.initial_learner_state(params)

    print(f"Training agent for {train_episodes} episodes")
    for episode in range(train_episodes):

        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        accumulator.push(timestep, None)
        actor_state = agent.initial_actor_state()

        while not timestep.last():

            # Acting.
            actor_output, actor_state = agent.actor_step(
                params, timestep, actor_state, next(rng), evaluation=False)

            # Agent-environment interaction.
            action = int(actor_output.actions)
            timestep = environment.step(action)

            # Accumulate experience.
            accumulator.push(timestep, action)

            # Learning.
            if accumulator.is_ready(batch_size):
                params, learner_state = agent.learner_step(
                    params, accumulator.sample(batch_size), learner_state, next(rng))  # type: ignore

        # Evaluation.
        if not episode % evaluate_every:
            returns = 0.
            for _ in range(eval_episodes):
                timestep = environment.reset()
                actor_state = agent.initial_actor_state()

                while not timestep.last():
                    actor_output, actor_state = agent.actor_step(
                        params, timestep, actor_state, next(rng), evaluation=True)
                    timestep = environment.step(int(actor_output.actions))
                    returns += timestep.reward

            avg_returns = returns / eval_episodes
            print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f}")


def main(unused_arg):
    ldba = JaxLDBA(
        num_states=1,
        num_actions=1,
        conditions=jp.array([True])
    )
    base_env = catch.Catch(seed=FLAGS.seed)

    env = LCRLEnv(
        ldba,
        base_env
    )
    epsilon_cfg = dict(
        init_value=FLAGS.epsilon_begin,
        end_value=FLAGS.epsilon_end,
        transition_steps=FLAGS.epsilon_steps,
        power=1.)
    agent = DQN(
        observation_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        epsilon_cfg=epsilon_cfg,
        target_period=FLAGS.target_period,
        learning_rate=FLAGS.learning_rate,
    )

    accumulator = ReplayBuffer(FLAGS.replay_capacity)
    run_loop(
        agent=agent,
        environment=env,
        accumulator=accumulator,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        train_episodes=FLAGS.train_episodes,
        evaluate_every=FLAGS.evaluate_every,
        eval_episodes=FLAGS.eval_episodes,
    )


if __name__ == "__main__":
    app.run(main)
