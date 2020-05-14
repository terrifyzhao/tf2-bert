import tensorflow as tf
import numpy as np
import six


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


class Train(object):

    def __init__(self,
                 epochs,
                 lr=1e-5,
                 optimizer=None,
                 is_binary=True,
                 debug=True):
        self.is_binary = is_binary
        self.epochs = epochs
        self.lowest = 1e10
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(lr)
        else:
            self.optimizer = optimizer
        if debug:
            tf.config.experimental_run_functions_eagerly(True)
        else:
            tf.config.experimental_run_functions_eagerly(False)
        if is_binary:
            self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
            self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
        else:
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def train(self, model, train_data, test_data=None, model_name='model'):

        @tf.function
        def train_step(inputs, labels):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = self.loss(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(labels, predictions)

        @tf.function
        def eval_step(inputs, labels):
            predictions = model(inputs)
            t_loss = self.loss(labels, predictions)
            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)

        train_iter = train_data.__iter__()
        if test_data is not None:
            test_iter = test_data.__iter__()
        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            template = 'Epoch {}, Step {}/{}, Loss: {}, Accuracy: {}'
            # train
            for step in range(train_data.steps):
                x, y = next(train_iter)

                train_step(x, y)
                print(template.format(epoch + 1,
                                      step + 1,
                                      train_data.steps,
                                      self.train_loss.result(),
                                      self.train_accuracy.result() * 100))

            # if step != 0 and (step + 1) % 20 == 0:
                # test
            if test_data is not None:
                for s in range(test_data.steps):
                    x, y = next(test_iter)
                    eval_step(x, y)
                print(
                    f'Test Loss: {self.test_loss.result()}, Test Accuracy: {self.test_accuracy.result() * 100}')

            print()

            # save model
            if self.lowest > self.train_loss.result():
                model.save_weights(model_name)
