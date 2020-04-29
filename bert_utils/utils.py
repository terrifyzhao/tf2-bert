import tensorflow as tf


def mask_op(x, mask, mode='mul'):
    if mask is None:
        return x
    else:
        if mode == 'mul':
            mask = tf.expand_dims(mask, 2)
            return x * mask
        elif mode == 'add':
            mask_a = tf.expand_dims(mask, 1)
            mask_b = tf.expand_dims(mask, 2)
            mask = mask_a * mask_b
            mask = tf.expand_dims(mask, 1)
            return x - (1 - mask) * 1e10
        else:
            return x


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
