import tensorflow as tf


def mask_op(x, mask, mode='mul', key_len=None):
    if mask is None:
        return x
    else:
        mask = tf.expand_dims(mask, -1)
        if mode == 'mul':
            return x * mask
        elif mode == 'add' and key_len is not None:
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, 1, 1, key_len])
            return x - (1 - mask) * 1e10
        else:
            return x


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
