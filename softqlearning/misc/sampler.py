import numpy as np
import time

from . import logger


def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    dones = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):
        action, agent_info = policy.get_action(observation)
        next_obs, reward, done, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        dones[t] = done
        rewards[t] = reward
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if done:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'dones': dones[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = list()
    for i in range(n_paths):
        paths.append(rollout(env, policy, path_length))

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_replay_buffer_size, batch_size):
        self._max_path_length = max_path_length
        self._min_replay_buffer_size = min_replay_buffer_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.replay_buffer = None

    def initialize(self, env, policy, replay_buffer):
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.replay_buffer.size >= self._min_replay_buffer_size
        return enough_samples

    def random_batch(self):
        return self.replay_buffer.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('replay_buffer-size', self.replay_buffer.size)


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, _ = self.policy.get_action(self._current_observation)
        next_observation, reward, done, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.replay_buffer.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            done=done,
            next_observation=next_observation)

        if done or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


class DummySampler(Sampler):
    def __init__(self, batch_size, max_path_length):
        super(DummySampler, self).__init__(
            max_path_length=max_path_length,
            min_replay_buffer_size=0,
            batch_size=batch_size)

    def sample(self):
        pass
