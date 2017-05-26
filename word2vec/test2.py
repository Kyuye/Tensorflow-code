import tensorflow as tf

number = tf.Variable("3")
hello = tf.constant("hello world")

with tf.Session() as sess:
  
    print(sess.run(hello))
