
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib import legacy_seq2seq

tf.reset_default_graph()
sess = tf.InteractiveSession()

seq_length = 5
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100

encode_input = [tf.placeholder(tf.int32, shape=(None,), name="inp%i"%t) for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i"%t) for t in range(seq_length)]
weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

decode_input = ([tf.zeros_like(encode_input[0], dtype=np.int32, name="GO")] + encode_input[:-1])

previous_memory = tf.zeros((batch_size, memory_dim))

cell = core_rnn_cell.GRUCell(memory_dim)

decode_outputs, decode_memory = legacy_seq2seq.embedding_rnn_seq2seq(encode_input, decode_input, cell, vocab_size, vocab_size, embedding_dim)

loss = legacy_seq2seq.sequence_loss(decode_outputs, labels, weights, vocab_size)
tf.summary.scalar("loss", loss)

manitude = tf.sqrt(tf.reduce_sum(tf.square(decode_memory[1])))
tf.summary.scalar("manitude at t=1", manitude)

summary_op = tf.summary.merge_all()

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)

summary_writer = tf.summary.FileWriter("./sources/seq2seq/log/", sess.graph)

sess.run(tf.global_variables_initializer())


def train_batch(batch_size):
    # X = [np.random.choice(vocab_size, size=(seq_length,), replace=False) for _ in range(batch_size)]
    X = [np.random.choice(vocab_size, (seq_length,), False) for _ in range(batch_size)]
    Y = X[:]

    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {encode_input[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary

for t in range(500):
    loss_t, summary = train_batch(batch_size)
    summary_writer.add_summary(summary, t)
summary_writer.flush()


