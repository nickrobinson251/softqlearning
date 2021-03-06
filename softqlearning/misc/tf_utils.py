import tensorflow as tf


def get_default_session():
    return tf.get_default_session() or create_session()


def create_session(**kwargs):
    """ Create new tensorflow session with given configuration. """
    if "config" not in kwargs:
        # kwargs["config"] = get_configuration()
        kwargs["config"] = None
    return tf.InteractiveSession(**kwargs)


def get_configuration():
    """ Returns personal tensorflow configuration. """
    config_args = {}
    return tf.ConfigProto(**config_args)
