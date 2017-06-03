
import tensorflow as tf
import matplotlib.pyplot as plt

array = list(range(-100, 100))
print(array[:10])
x1 = 0.01 * tf.constant(array, tf.float32)
x2 = tf.sqrt(1-tf.square(x1))

with tf.Session() as sess:
    _x1, _x2 = sess.run([x1, x2])
    print(_x1[:10])
    print(_x2[:10])

plt.scatter(_x1, _x2)
plt.scatter(_x1, -_x2)
plt.show()