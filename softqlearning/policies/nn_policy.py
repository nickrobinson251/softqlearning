import tensorflow as tf

from softqlearning.misc import Parameterized, Serializable, tf_utils


class Policy(Parameterized):
    def __init__(self):
        Parameterized.__init__(self)

    def get_action(self, observation):
        raise NotImplementedError

    def get_actions(self, observations):
        raise NotImplementedError

    def reset(self, dones=None):
        pass


class NNPolicy(Policy, Serializable):
    def __init__(self, env, obs_pl, action, scope_name=None):
        Serializable.quick_init(self, locals())
        self._obs_pl = obs_pl
        self._action = action
        self._scope_name = (tf.get_variable_scope().name
                            if not scope_name else scope_name)
        super(NNPolicy, self).__init__()

    def get_action(self, observation):
        return self.get_actions(observation[None])[0], None

    def get_actions(self, observations):
        feeds = {self._obs_pl: observations}
        actions = tf_utils.get_default_session().run(self._action, feeds)
        return actions

    def log_diagnostics(self, paths):
        pass

    def get_params_internal(self, **tags):
        # TODO: rewrite this using tensorflow collections
        if tags:
            raise NotImplementedError
        scope = self._scope_name
        # Add "/" to 'scope' unless it's empty (otherwise get_collection will
        # return all parameters that start with 'scope'.
        scope = scope if scope == '' else scope + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
