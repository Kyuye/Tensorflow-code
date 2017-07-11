import tensorflow as tf

const = tf.constant("hello tokki")

with tf.Session() as sess:
    print(sess.run(const))

    



