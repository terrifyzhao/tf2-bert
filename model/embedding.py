import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *


class Embedding(Model):
    def __init__(self):
        super(Embedding,self).__init__()

        full_position_embeddings = tf.get_variable(
            name=position_embedding_name,
            shape=[max_position_embeddings, width],
            initializer=create_initializer(initializer_range))
        # Since the position embedding table is a learned variable, we create it
        # using a (long) sequence length `max_position_embeddings`. The actual
        # sequence length might be shorter than this, for faster training of
        # tasks that do not have long sequences.
        #
        # So `full_position_embeddings` is effectively an embedding table
        # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
        # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
        # perform a slice.
        position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                       [seq_length, -1])
        num_dims = len(output.shape.as_list())

        # Only the last two dimensions are relevant (`seq_length` and `width`), so
        # we broadcast among the first dimensions, which is typically just
        # the batch size.
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([seq_length, width])
        position_embeddings = tf.reshape(position_embeddings,
                                         position_broadcast_shape)
        output += position_embeddings

        output = layer_norm_and_dropout(output, dropout_prob)
        return output
