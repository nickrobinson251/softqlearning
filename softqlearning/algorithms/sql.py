import numpy as np
import tensorflow as tf

from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.misc import logger, tf_utils

from .rl_algorithm import RLAlgorithm

EPS = np.finfo(np.float32).eps


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class SQL(RLAlgorithm):
    """Soft Q-learning algorithm (SQL).

    Reference:
        [1] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine,
        "Reinforcement Learning with Deep Energy-Based Policies," International
        Conference on Machine Learning, 2017. https://arxiv.org/abs/1702.08165

    ...

    Parameters
    ----------
    sampler : Sampler
        Sampler instance to use for sampling episodes from replay buffer
        (ReplayBuffer nstance supplied at train time)
    env : gym.Env:
        gym environment object
    replay_buffer : ReplayBuffer
        Replay buffer to add gathered samples to
    q_function : NNQFunction
        Q-function approximator
    policy : NNPolicy
        A policy function approximator
    discount : float (default=0.99)
        Discount factor gamma
    epoch_length : int (default=1000)
    eval_n_episodes : int (default=10)
        Number of rollouts to evaluate
    eval_render : bool (default=False)
        Whether or not to render the evaluation environment
    kernel_fn : function object
        A function object that represents a kernel function
    kernel_n_particles : int (default=16)
        Number of particles per state used in SVGD updates
    kernel_update_ratio : float (defualt=0.5)
        Ratio of SVGD particles used forcomputating inner/outer empirical
        expectation
    n_epochs : int (default=1000)
        Number of epochs to run training
    n_train_repeat : int (default=1)
        Number of times to repeat the training for single time step
    plotter : Plotter (default=None)
        Plotter instance used for visualizing Q-function during training
    policy_lr : float (default=1e-3)
        Learning rate for the policy approximator
    q_function_lr : float (default=1e-3)
        Learning rate for the Q-function approximator
    reward_scale : float (defualt=1)
        factor to scale the raw rewards; useful for adjusting the
        temperature of the optimal Boltzmann
        distribution.
    save_full_state : bool (default=True)
        If true, full algorithm state and replay buffer are saved
    td_target_update_interval : int (default=1)
        How often to update target network to match the current Q-function
    train_policy : bool (default=True)
        If True, policy gets trained
    train_qf : bool
        If True, Q-function getss trained
    use_saved_qf : bool (default=False)
        If true, initial parameters in the provided Q-function are used;
        if False, Q-function parameters are reinitialised
    use_saved_policy : bool (default=False)
        If true, initial parameters in the provided policy are used;
        if False, policy parameters are reinitialised
    value_n_particles : int (default=16)
        Number of action samples used for estimating the value of next state
    """

    def __init__(
            self,
            sampler,
            env,
            replay_buffer,
            q_function,
            policy,
            discount=0.99,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_render=False,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            n_epochs=1000,
            n_train_repeat=1,
            plotter=None,
            policy_lr=1e-3,
            q_function_lr=1e-3,
            reward_scale=1,
            save_full_state=False,
            td_target_update_interval=1,
            train_policy=True,
            train_qf=True,
            use_saved_policy=False,
            use_saved_qf=False,
            value_n_particles=16):
        self.env = env
        self.plotter = plotter
        self.policy = policy
        self.q_function = q_function
        self.replay_buffer = replay_buffer
        self._action_dim = self.env.action_space.n
        self._discount = discount
        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio
        self._observation_dim = np.prod(self.env.observation_space.shape)
        self._policy_lr = policy_lr
        self._q_function_lr = q_function_lr
        self._qf_target_update_interval = td_target_update_interval
        self._reward_scale = reward_scale
        self._save_full_state = save_full_state
        self._sess = tf_utils.get_default_session()
        self._target_ops = []
        self._train_policy = train_policy
        self._train_qf = train_qf
        self._training_ops = []
        self._value_n_particles = value_n_particles
        super(SQL, self).__init__(**dict(
            sampler=sampler,
            epoch_length=epoch_length,
            eval_n_episodes=eval_n_episodes,
            eval_render=eval_render,
            n_epochs=n_epochs,
            n_train_repeat=n_train_repeat))

        self._create_placeholders()
        self._create_svgd_update()
        self._create_target_ops()
        self._create_td_update()

        self._sess.run(tf.global_variables_initializer())

        if use_saved_qf:
            saved_qf_params = q_function.get_param_values()
            self.q_function.set_param_values(saved_qf_params)
        if use_saved_policy:
            saved_policy_params = policy.get_param_values()
            self.policy.set_param_values(saved_policy_params)

    def _create_placeholders(self):
        """Create placeholders for observations, actions, rewards, and dones."""
        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations')

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='next_observations')

        self._actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='actions')

        self._next_actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='next_actions')

        self._rewards_ph = tf.placeholder(
            tf.float32, shape=[None], name='rewards')

        self._dones_ph = tf.placeholder(
            tf.float32, shape=[None], name='dones')

    def _create_td_update(self):
        """Create a minimization operation for Q-function update."""
        with tf.variable_scope('target'):
            # The value of the next state is approximated with uniform samples.
            target_actions = tf.random_uniform(
                shape=(1, self._value_n_particles, self._action_dim),
                minval=-1,
                maxval=1)
            q_value_targets = self.q_function.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=target_actions)
            assert_shape(q_value_targets, [None, self._value_n_particles])

        self._q_values = self.q_function.output_for(
            observations=self._observations_ph,
            actions=self._actions_ph,
            reuse=True)
        assert_shape(self._q_values, [None])

        # Equation 10: V(s_t) = alpha * log E[ exp(Q(s_t, a') / alpha) / q(a') ]
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        assert_shape(next_value, [None])

        # Importance weights add just a constant to the value.
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))
        next_value += self._action_dim * np.log(2)

        # \hat Q in Equation 11: hatQ(s_t, a_t) = r_t + gamma * E[ V(s_{t+1} ]
        ys = tf.stop_gradient(
            (self._reward_scale * self._rewards_ph
             + (1 - self._dones_ph) * self._discount * next_value))
        assert_shape(ys, [None])

        # Equation 11: J(theta) = E[ (hatQ(s_t, a_t) - q(s_t, a_t))^2 / 2 ]
        self._bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)

        if self._train_qf:
            td_train_op = tf.train.AdamOptimizer(self._q_function_lr).minimize(
                loss=self._bellman_residual,
                var_list=self.q_function.get_params_internal())
            self._training_ops.append(td_train_op)

    def _create_svgd_update(self):
        """Create a minimization operation for policy update (SVGD)."""
        actions = self.policy.actions_for(
            observations=self._observations_ph,
            n_action_samples=self._kernel_n_particles,
            reuse=True)
        assert_shape(actions,
                     [None, self._kernel_n_particles, self._action_dim])

        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        n_updated_actions = int(self._kernel_n_particles
                                * self._kernel_update_ratio)
        n_fixed_actions = self._kernel_n_particles - n_updated_actions

        fixed_actions, updated_actions = tf.split(
            actions,
            [n_fixed_actions, n_updated_actions],
            axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)
        assert_shape(fixed_actions, [None, n_fixed_actions, self._action_dim])
        assert_shape(updated_actions,
                     [None, n_updated_actions, self._action_dim])

        svgd_target_values = self.q_function.output_for(
            observations=self._observations_ph[:, None, :],
            actions=fixed_actions,
            reuse=True)

        # Target log-density. Q_soft in Equation 13:
        squash_correction = tf.reduce_sum(
            tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction
        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        assert_shape(grad_log_p, [None, n_fixed_actions, 1, self._action_dim])

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = tf.expand_dims(kernel_dict["output"], dim=3)
        assert_shape(kappa, [None, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel_dict["gradient"],
            reduction_indices=1)
        assert_shape(action_gradients,
                     [None, n_updated_actions, self._action_dim])

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tf.gradients(
            ys=updated_actions,
            xs=self.policy.get_params_internal(),
            grad_ys=action_gradients)

        surrogate_loss = tf.reduce_sum(
            [tf.reduce_sum(w * tf.stop_gradient(g))
             for w, g in zip(self.policy.get_params_internal(), gradients)])

        if self._train_policy:
            optimizer = tf.train.AdamOptimizer(self._policy_lr)
            svgd_training_op = optimizer.minimize(
                loss=-surrogate_loss,
                var_list=self.policy.get_params_internal())
            self._training_ops.append(svgd_training_op)

    def _create_target_ops(self):
        """Create tensorflow operation for updating the target Q-function."""
        if not self._train_qf:
            return

        source_params = self.q_function.get_params_internal()
        target_params = self.q_function.get_params_internal(scope='target')

        self._target_ops = [
            tf.assign(tg, src) for tg, src in zip(target_params, source_params)]

    # TODO: do not pass, policy, and replay_buffer to `__init__` directly.
    def train(self):
        self._train(self.env, self.policy, self.replay_buffer)

    def _init_training(self):
        self._sess.run(self._target_ops)

    def _do_training(self, iteration, batch):
        """Run the operations for updating training and target ops."""
        feed_dict = self._get_feed_dict(batch)
        self._sess.run(self._training_ops, feed_dict)

        if iteration % self._qf_target_update_interval == 0 and self._train_qf:
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""
        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._dones_ph: batch['dones'],
        }
        return feed_dict

    def log_diagnostics(self, batch):
        """Record diagnostic information.

        Records the mean and standard deviation of Q-function and the
        squared Bellman residual of the s (mean squared Bellman error)
        for a sample batch.

        Also call the `draw` method of the plotter, if plotter is defined.
        """
        feeds = self._get_feed_dict(batch)
        q_function, bellman_residual = self._sess.run(
            [self._q_values, self._bellman_residual], feeds)

        logger.record_tabular('q_function-avg', np.mean(q_function))
        logger.record_tabular('q_function-std', np.std(q_function))
        logger.record_tabular('mean-sq-bellman-error', bellman_residual)

        self.policy.log_diagnostics(batch)
        if self.plotter:
            self.plotter.draw()

    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SQL algorithm.

        If `self._save_full_state == True`, returns snapshot including the
        replay buffer. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, and environment instances.
        """
        snapshot = {
            'epoch': epoch,
            'policy': self.policy,
            'q_function': self.q_function,
            'env': self.env}
        if self._save_full_snapshot:
            snapshot['replay_buffer'] = self.replay_buffer
        return snapshot
