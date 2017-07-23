
import tensorflow as tf


dict_n = {'a':1, 'b':2, 'c':3, 'd':4}
for i in dict_n:
    print(i)

# c1 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
# c2 = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 2])

# m = tf.matmul(c1, c2)

# with tf.Session() as sess:
#     print(sess.run(m))


# filename = os.path.join(os.getcwd(), "DataSet/twitter_emotion_v2(p,n,N).csv")
# queue = tf.train.string_input_producer([filename])

# reader = tf.TextLineReader(skip_header_lines=1)
# key, value = reader.read(queue)

# _, sent, _, c0, c1, c2, c3, c4, c5, c6, c7 = tf.decode_csv(value, [[""]]*11)
# content = tf.stack([c0, c1, c2, c3, c4, c5, c6, c7])

# label, batch = tf.train.batch([sent, content], 10)
# batch = tf.expand_dims(tf.reduce_join(batch, 1), axis=1)
# label = tf.expand_dims(label, axis=1)


# filename = os.getcwd() + "/dataset/twitter_emotion_v2(p,n,N).csv"
# data = pandas.read_csv(filename, usecols=["Sentiment", "content"], nrows=100)

# print(data)

# filename_queue = tf.train.string_input_producer([FLAGS.train_data])

# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)

# _, sentiment, _, c1, c2, c3, c4, c5, c6, c7, c8 = \
# tf.decode_csv(value, [[""]]*11)

# contents = tf.stack([c1, c2, c3, c4, c5, c6, c7, c8])

# label, batch = tf.train.batch([sentiment, contents], 10)
# batch = tf.expand_dims(tf.reduce_join(batch, 1), axis=1)
# label = tf.expand_dims(label, axis=1)

# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     thread = tf.train.start_queue_runners(sess, coord)
#     print()
#     _label, train = sess.run([label, batch])
#     print(_label)
#     print(train)
    
#     coord.request_stop()
#     coord.join(thread)




# filename_queue = tf.train.string_input_producer(["gs://wgan/dataset/twitter_emotion_v2(p,n,N).csv"])

# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)

# data = tf.decode_csv(value, [[""]]*11)

# batch = tf.train.batch(data, 10)

# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     thread = tf.train.start_queue_runners(sess, coord)
#     for _ in range(10):
#         print(sess.run(batch))
    
#     coord.request_stop()
#     coord.join(thread)