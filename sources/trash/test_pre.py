from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import nltk, re, pprint

import tensorflow as tf
import pandas
import json
import os
import numpy as np
import re 
from pandas import Series, DataFrame 
import string 
 

pandas.set_option('display.max_colwidth', 10000)

filename = "./dataset/twitter_emotion_v2(p,n,N).csv"
        
data = pandas.read_csv(filename, usecols=["Sentiment", "content"])
data = data[data["content"] != "0"]
data["content"] = data["content"].astype("str")
sent1 = data["content"]

sent2 = sent1.str.split(" ")

print(sent2)
print()
print()


def filter_word(target,sent):
    filtered = sent
    for remove in target:
         filtered = list(map(lambda s: filter(lambda w: w.find(remove), filter(None, s)), filtered))
    return filtered

def filter_printable(sent):
    printable_set = set(string.printable)
    return list(map(lambda s: filter(lambda w: w[0] in printable_set, s), sent))

def filter_none(sent):
    return filter(None, sent)
# print(list(map(lambda sent: filter(lambda w: w[0] in printable, sent), afiltered)))

remove_words = ["@", "http://", "--", ":", ";","&","=",">","<","$","~","#","//","^", "(", ")", ",","\\"]
test = filter_none(sent2)
test = filter_word(remove_words,test)
test = filter_printable(test)

print(test)









# df = pandas.DataFrame(sent1)
# pandas.Series(sent1).str.replace('.','')
# sent2 = pandas.Series(sent1).str.replace('@',' ')
# print(pandas.Series(sent1).str.replace('http://',''))


# print(re.sub(r'[^\x00-\x7F]+',' ', sent2))
# pandas.Series(sent1).str.replace('http://','')
# print(sent1)
# a = df.columns.str.replace('@','')
# print(sent1)
# print(pandas.Series(sent1).str.replace('@',''))
# ".@http://www.com","",sent1.str)
# aa = re.sub("asd", "", "asdfg123")
# print(sent1.find('@'))
# print(sent1)
# sent2 = sent1.str.replace('.', ' ')
# print(aa)




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