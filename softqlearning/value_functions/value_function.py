import tensorflow as tf
import numpy as np

from softqlearning.misc.nn import MLPFunction
from softqlearning.misc import Serializable, tf_utils


class NNVFunction(MLPFunction):
    def __init__(self,
                 env,
                 hidden_layer_sizes=(128, 128),
                 name='value_function'):

        self._Do = np.prod(env.observation_space.shape)
        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._Do], name='observations')

        Serializable.quick_init(self, locals())
        super(NNVFunction, self).__init__(
            inputs=(self._observations_ph, ),
            name=name,
            hidden_layer_sizes=hidden_layer_sizes)

    def eval(self, observations):
        return super(NNVFunction, self)._eval((observations, ))

    def output_for(self, observations, reuse=False):
        return super(NNVFunction, self)._output_for(
            (observations, ), reuse=reuse)


class NNQFunction(MLPFunction):
    def __init__(self,
                 env,
                 hidden_layer_sizes=(128, 128),
                 name='q_function'):

        self._Da = env.action_space.n
        self._Do = np.prod(env.observation_space.shape)

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._Do], name='observations')
        self._actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._Da], name='actions')

        Serializable.quick_init(self, locals())
        super(NNQFunction, self).__init__(
            inputs=(self._observations_ph, self._actions_ph),
            name=name,
            hidden_layer_sizes=hidden_layer_sizes)

    def output_for(self, observations, actions, reuse=False):
        return super(NNQFunction, self)._output_for(
            (observations, actions), reuse=reuse)

    def eval(self, observations, actions):
        return super(NNQFunction, self)._eval((observations, actions))


class SumQFunction(Serializable):
    def __init__(self, env, q_functions):

        self.q_functions = q_functions

        self._Da = env.action_space.n
        self._Do = np.prod(env.observation_space.shape)

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._Do], name='observations')
        self._actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._Da], name='actions')

        self._output = self.output_for(
            self._observations_ph, self._actions_ph, reuse=True)
        Serializable.quick_init(self, locals())

    def output_for(self, observations, actions, reuse=False):
        outputs = [
            qf.output_for(observations, actions, reuse=reuse)
            for qf in self.q_functions
        ]
        output = tf.add_n(outputs)
        return output

    def _eval(self, observations, actions):
        feeds = {
            self._observations_ph: observations,
            self._actions_ph: actions
        }

        return tf_utils.get_default_session().run(self._output, feeds)

    def get_param_values(self):
        all_values_list = [qf.get_param_values() for qf in self.q_functions]

        return np.concatenate(all_values_list)

    def set_param_values(self, all_values):
        param_sizes = [qf.get_param_values().size for qf in self.q_functions]
        split_points = np.cumsum(param_sizes)[:-1]

        all_values_list = np.split(all_values, split_points)

        for values, qf in zip(all_values_list, self.q_functions):
            qf.set_param_values(values)
