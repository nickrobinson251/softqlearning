import abc
import gtimer as gt
import numpy as np

from softqlearning.misc import logger
from softqlearning.misc.sampler import rollouts
from softqlearning.misc.utils import deep_clone


class RLAlgorithm:
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.

    ...

    Parameters
    ----------
    sampler : Sampler
        Sampler instance to use for sampling episodes from replay buffer
        (ReplayBuffer instance supplied at train time)
    epoch_length : int (default=1000)
    eval_n_episodes : int (default=10)
        Number of rollouts to evaluate
    eval_render : bool (default=False)
        Whether or not to render the evaluation environment
    n_epochs : int (default=1000)
        Number of epochs to run training
    n_train_repeat : int (default=1)
        Number of times to repeat the training for single time step
    """

    def __init__(
            self,
            sampler,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_render=False,
            n_epochs=1000,
            n_train_repeat=1):
        self.sampler = sampler

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length

        self._eval_n_episodes = eval_n_episodes
        self._eval_render = eval_render

        self.env = None
        self.policy = None
        self.replay_buffer = None

    def _train(self, env, policy, replay_buffer, sess):
        """Perform RL training.

        Parameters
        ----------
        env : gym.Env
            Environment used for training
        policy : Policy
            Policy used for training
        replay_buffer : ReplayBuffer
            Replay buffer to add samples to
        """
        self._init_training()
        self.sampler.initialize(env, policy, replay_buffer)
        evaluation_env = deep_clone(env) if self._eval_n_episodes else None

        with sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(
                    range(self._n_epochs + 1), save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    self.sampler.sample()
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            iteration=t + epoch*self._epoch_length,
                            batch=self.sampler.random_batch())
                    gt.stamp('train')

                self._evaluate(policy, evaluation_env)
                gt.stamp('eval')

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)

                time_itrs = gt.get_times().stamps.itrs
                time_eval = time_itrs['eval'][-1]
                time_total = gt.get_times().total
                time_train = time_itrs.get('train', [0])[-1]
                time_sample = time_itrs.get('sample', [0])[-1]

                logger.record_tabular('time-train', time_train)
                logger.record_tabular('time-eval', time_eval)
                logger.record_tabular('time-sample', time_sample)
                logger.record_tabular('time-total', time_total)
                logger.record_tabular('epoch', epoch)

                logger.record_time(time_total, time_sample)

                self.sampler.log_diagnostics()

                logger.save_stats()
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

            self.sampler.terminate()

    def _evaluate(self, policy, evaluation_env):
        """Perform evaluation for the current policy."""
        if self._eval_n_episodes < 1:
            return

        # TODO: max_episode_length should be a property of environment.
        episodes = rollouts(
            evaluation_env,
            policy,
            episode_length=self.sampler._max_episode_length,
            n_episodes=self._eval_n_episodes)
        total_returns = [episode['rewards'].sum() for episode in episodes]
        episode_lengths = [len(episode['rewards']) for episode in episodes]

        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))

        logger.record_returns(total_returns)

        evaluation_env.log_diagnostics(episodes)
        if self._eval_render:
            evaluation_env.render(episodes)

        if self.sampler.batch_ready():
            batch = self.sampler.random_batch()
            self.log_diagnostics(batch)

    @abc.abstractmethod
    def log_diagnostics(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self):
        raise NotImplementedError
