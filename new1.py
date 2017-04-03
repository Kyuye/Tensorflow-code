import tensorflow as tf

const = tf.constant("hello world")

with tf.Session() as sess:
    print(sess.run(const))