
import tensorflow as tf
import matplotlib.pyplot as plt

def is_prime(n):
    if n == 0 or n == 1:
        return False, -1

    for i in range(2, n-1):
        if n % i == 0:
            return False, -1 
    return True, n

s = []
for i in range(100000):
    is_num, num = is_prime(i)
    if is_num: 
        s.append(num)

x = tf.constant(list(range(len(s))), tf.float32)
y = 9.0 * x - 1000 

with tf.Session() as sess:
    p = sess.run(y)

plt.plot(s)
plt.plot(p)
plt.savefig("./prime_plot.png")