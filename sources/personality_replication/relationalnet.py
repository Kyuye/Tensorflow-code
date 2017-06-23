
import tensorflow as tf
import collections

ids = tf.constant([[1, 5], [2, 5], [2, 5], [0, 4], [0, 4], [3, 4]])
vectors = tf.contrib.layers.embed_sequence(
    ids=ids,
    vocab_size=6,
    embed_dim=2
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(vectors))

exit()

vocabulary_size=50000
state_size = 32
q_state_size = 32
regularizer_scale=0.5
g_hidden1 = 256
g_hidden2 = 256
g_hidden3 = 256
g_hidden4 = 256
g_hidden5 = 256
g_hidden6 = 256
g_class = 256

f_hidden1 = 256
f_hidden2 = 256
f_hidden3 = 256
f_class = vocabulary_size

words = ["all", "of", "the", "words", "in", "the", "files"]
word_counts = collections.Counter(words).most_common(vocabulary_size)

word_ids = {"UNK":0}
for w, count in word_counts:
    word_ids[w] = len(word_ids)

sentence1 = ["one", "of", "the", "sentence", "in", "the", "train", "data"]
sentence2 = ["one", "of", "the", "sentence", "in", "the", "train", "data"]
question = ["one", "of", "the", "question", "in", "the", "train", "data"]

fact1_ids = list(map(lambda w: word_ids[w]), sentence1)
fact2_ids = list(map(lambda w: word_ids[w]), sentence2)
question_ids = list(map(lambda w: word_ids[w]), question)

fact1_embeds = tf.contrib.layers.embed_sequence(fact1_ids)
fact2_embeds = tf.contrib.layers.embed_sequence(fact2_ids)
question_embeds = tf.contrib.layers.embed_sequence(question_ids)

with tf.variable_scope("object") as scope:
    o_lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size, dtype=tf.float32)
    _, o1 = tf.nn.dynamic_rnn(o_lstm_cell, fact1_embeds)
    scope.reuse_variables()
    _, o2 = tf.nn.dynamic_rnn(o_lstm_cell, fact2_embeds)
with tf.variable_scope("question"):
    q_lstm_cell = tf.contrib.rnn.BasicLSTMCell(q_state_size, dtype=tf.float32)
    _, q = tf.nn.dynamic_rnn(q_lstm_cell, question_embeds)

g_in = tf.concat([o1, o2, q], axis=0)

g_layer1 = tf.layers.dense(
    inputs=g_in,
    units=g_hidden1,
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    trainable=True,
    name="g_layer1"
)

g_layer2 = tf.layers.dense(
    inputs=g_layer1,
    units=g_hidden2,
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    trainable=True,
    name="g_layer2"
)

g_layer3 = tf.layers.dense(
    inputs=g_layer2,
    units=g_hidden3,
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    trainable=True,
    name="g_layer3"
)

g_out = tf.layers.dense(
    inputs=g_layer3,
    units=g_class,
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    trainable=True,
    name="g_out"
)

f_in = tf.reduce_sum(g_out, axis=0)

f_layer1 = tf.layers.dense(
    inputs=f_in,
    units=f_hidden1,
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    trainable=True,
    name="f_layer1"
)

f_layer2 = tf.layers.dense(
    inputs=f_layer1,
    units=f_hidden2,
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    trainable=True,
    name="f_layer2"
)

f_out = tf.layers.dense(
    inputs=f_layer2,
    units=f_class,
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
    trainable=True,
    name="f_out"
)

# const1 = tf.constant([1, 2, 3, 4, 5])
# const2 = tf.constant([6, 7, 8, 9, 10])

# merge = tf.concat([const1, const2], axis=0)

# with tf.Session() as sess:
#     print(sess.run(merge))