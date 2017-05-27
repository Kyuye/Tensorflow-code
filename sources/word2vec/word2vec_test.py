
import tensorflow as tf
from functools import reduce
import os
import csv
import collections
import json
from pprint import pprint
import matplotlib.pyplot as plt


file_dir = "./DataSet/"

def csv_to_text(file_dir):
    filelist = [file_dir+i for i in filter(lambda x: x.rfind(".csv") > 0, os.listdir(file_dir))]

    with open(file_dir+"songdata.csv", 'r') as f:
        reader = csv.reader(f)
        data = list(map(lambda x: x[3] , reader))
        del data[0]
        
    with open(file_dir+"songdata.txt", 'w') as f:
        for i in data:
            f.writelines(i)

def words_read_text(file_dir):
    with open(file_dir+'songdata.txt', 'r')  as f:
        return [word for line in f for word in line.split()]

def word_count(file_dir):
    words = words_read_text(file_dir)
    return collections.Counter(words).most_common(100000)

def vocab_to_dict(file_dir):
    words = word_count(file_dir)
    vocab_dict = {}
    for w in words:
        idx = len(vocab_dict)
        vocab_dict[w[0]] = idx 

    return words, vocab_dict

def vocab_write_json(file_dir):
    vocab_dict = vocab_to_dict(file_dir)
    with open(file_dir+'vocab_dict.txt', 'w') as outfile:
        json.dump(vocab_dict, outfile)

def vocab_read_json():
    with open(file_dir+'vocab_dict.txt', 'r') as f:
        data = json.load(f)
    return data


def build_train_data(words_id, skip_window, data_num): 
    batch = [[]]
    for i in range(skip_window, data_num-skip_window+1):
        for n in range(i-skip_window, i+skip_window+1):
            if i == n:
                continue
            batch += [[words_id[i], words_id[n]]] 
        if i == skip_window+1:
            del batch[0]
    return batch

def write_train_data(batch):
    with open("./DataSet/train_set.csv", 'w') as f:
        writer = csv.writer(f, "excel")
        for row in batch:
            writer.writerow(row)


words_pair, vocab_dict = vocab_to_dict(file_dir)
data = words_read_text(file_dir)
words = [i[0] for i in words_pair]
words_id = [vocab_dict[i] for i in data[:100]]


batch = build_train_data(words_id, 2, 20)
write_train_data(batch)

exit()

filename = file_dir + 'songdata.txt'
filename_queue = tf.train.string_input_producer([filename])

reader = tf.TextLineReader()

key, value = reader.read(filename_queue)

words = tf.string_split([value], " ")

word_set = tf.train.batch([words], 10)

for i in range(10):
    w = word_set.values[i]

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)

    print(sess.run(w))

    coord.request_stop()
    coord.join(thread)