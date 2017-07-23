
import tensorflow as tf
import pandas
import json
import os
import numpy as np
import re 
from utils import *
from pandas import Series, DataFrame 
import csv 

filename = "./dataset/twitter_emotion_v2(p,n,N).csv"


data = pandas.read_csv(filename, usecols=["Sentiment", "content"])
data = data[data["content"] != "0"]
data["content"] = data["content"].astype("str")
data["Sentiment"] = data["Sentiment"].astype("str")


remove_words = ["@", "http://", "--", ":", ";","&","=",">","<","$","~","#","//","^", "(", ")", ",","\\"]

        
        # with open(file_dir, 'r') as f:
        #     reader = csv.reader(f)
        # data = list(map(lambda x: x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10], reader))
            # del data[0]

__neg = data[data['Sentiment'].str.contains("Neg")]
_neg = __neg["content"]
__pos = data[data['Sentiment'].str.contains("Pos")]
_pos = __pos["content"]
__neu = data[data['Sentiment'].str.contains("neutral")]
_neu = __neu["content"]

neu = _neu.str.split(" ")
pos = _pos.str.split(" ")
neg = _neg.str.split(" ")

final_neu = filter_none(neu)
final_neu = filter_word(remove_words,final_neu)
final_neu = filter_printable(final_neu)

final_pos = filter_none(pos)
final_pos = filter_word(remove_words,final_pos)
final_pos = filter_printable(final_pos)

final_neg = filter_none(neg)
final_neg = filter_word(remove_words,final_neg)
final_neg = filter_printable(final_neg)


# print("Negative contents:", type(final_neg))
# print()
# print("Positive contents:", final_pos)
# print()
# print("Neutral contents:", final_neu)

# with open(file_dir[:-4]+".txt", 'w') as f:
#     f.write(str(final))

with open("./DataSet/Negative.csv", 'w') as f:
    writer = csv.writer(f, "excel")
    writer.writerow(final_neg)

with open("./DataSet/Neutral.csv", 'w') as f:
    writer = csv.writer(f, "excel")
    writer.writerow(final_neu)

with open("./DataSet/Positive.csv", 'w') as f:
    writer = csv.writer(f, "excel")
    writer.writerow(final_pos)
# print(sent1[df['Sentiment'].str.contains("Pos")])
# print(sent1[df['Sentiment'].str.contains("neutral")])

# pandas.Series(sent1).str.replace('.','')
# print(pandas.Series(sent1).str.replace('@',' '))
# pandas.Series(sent1).str.replace('http://','')


# print("Neg only :",select_rows(sent1,['Neg']))
# print("content only : ", sent1.filter(items=['content']))
# print("sentiment only:", sent1.filter(items=['Sentiment']))

# print(sent1.filter(like='Neg', axis= 0))

    
# ".@http://www.com","",sent1.str)
# aa = re.sub("asd", "", "asdfg123")
# print(sent1.find('@'))
# print(sent1)
# sent2 = sent1.str.replace('.', ' ')
# print(aa)

# df.to_csv('~/data/sample6.csv')



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