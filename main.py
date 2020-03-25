import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow import keras
from utils import *
from model.multihead import MultiHeadAttention

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train_x = train['question'].apply(seq2index)
train_x = padding_seq(train_x)
test_x = test['question'].apply(seq2index)
test_x = padding_seq(test_x)

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train['label'].values)).shuffle(1024).batch(256)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test['label'].values)).batch(256)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = MultiHeadAttention(12, 768)
        self.embedding = Embedding(411, 768)
        self.d1 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.layer([x, x, x])
        x = tf.reduce_mean(x, axis=1)
        x = self.d1(x)
        return x


model = MyModel()
loss_object = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

train_loss = keras.metrics.Mean(name='train_loss')
train_acc = keras.metrics.SparseCategoricalAccuracy(name='train_acc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

tf.config.experimental_run_functions_eagerly(True)


@tf.function
def train_step(texts, labels):
    with tf.GradientTape() as tape:
        predictions = model(texts)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions)


@tf.function
def test_step(texts, labels):
    predictions = model(texts)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_acc(labels, predictions)


epochs = 10
for epoch in range(epochs):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

    template = 'Epoch {}, step {}, Loss: {}, Accuracy: {}'
    for step, (texts, labels) in enumerate(train_ds):
        train_step(texts, labels)
        if step % 10 == 0 or len(texts) != 2048:
            print(template.format(step + 1,
                                  step,
                                  train_loss.result(),
                                  train_acc.result() * 100))

    template = 'Epoch {}, Test Loss: {}, Test Accuracy: {}'
    for texts, labels in test_ds:
        test_step(texts, labels)
    print(template.format(epoch + 1,
                          test_loss.result(),
                          test_acc.result() * 100))
