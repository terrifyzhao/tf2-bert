import tensorflow as tf


def mask_op(x, mask, mode='mul'):
    if mask is None:
        return x
    else:
        for _ in range(len(x.shape) - len(mask.shape)):
            mask = tf.expand_dims(mask, len(mask.shape))
        if mode == 'mul':
            return x * mask
        elif mode == 'add':
            return x - (1 - mask) * 1e10
        else:
            return x


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
