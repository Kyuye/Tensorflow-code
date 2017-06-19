
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import csv
import json
import collections
import matplotlib.pyplot as plt

file_dir = "./DataSet/"
data_name = file_dir + "twitter_emotion.csv"

class skip_gram(object):
    def __init__(self):
        self.vocabulary_size = 50000
        self.embed_size = 300
        self.num_sampled = 64
        self.batch_size = 32

    def csv_to_text(self, file_dir):
        with open(data_name, 'r') as f:
            reader = csv.reader(f)
            data = list(map(lambda x: x[3] , reader))
            del data[0]

        with open(data_name, 'w') as f:
            for i in data:
                f.writelines(i)

    def words_read_text(self, file_dir):
        with open(data_name, 'r')  as f:
            return [word for line in f for word in line.split()]

    def word_count(self, file_dir):
        words = self.words_read_text(file_dir)
        return collections.Counter(words).most_common(self.vocabulary_size)

    def vocab_to_dict(self, file_dir):
        words = self.word_count(file_dir)
        vocab_dict = {"UNK":0}
        for w in words:
            idx = len(vocab_dict)
            vocab_dict[w[0]] = idx 
        return words, vocab_dict


    def build_train_data(self, words_id, skip_window, data_num): 
        batch = [[]]
        for i in range(skip_window, data_num-skip_window+1):
            for n in range(i-skip_window, i+skip_window+1):
                if i == n:
                    continue
                batch += [[words_id[i], words_id[n]]] 
            if i == skip_window+1:
                del batch[0]
        return batch

    def write_train_data(self, batch):
        with open("./DataSet/train_set.csv", 'w') as f:
            writer = csv.writer(f, "excel")
            for row in batch:
                writer.writerow(row)


    def train_model(self):
        with tf.name_scope("train_set_build"):
            filename = file_dir + 'train_set.csv'
            filename_queue = tf.train.string_input_producer([filename])

            reader = tf.TextLineReader()

            key, value = reader.read(filename_queue)

            trains, labels = tf.decode_csv(value, [[0]]*2)
            train_batch, label_batch = tf.train.batch([trains, labels], self.batch_size)
            # train_batch = tf.reshape(train_batch, shape=(-1, 1))
            label_batch = tf.reshape(label_batch, shape=(-1, 1))

        with tf.name_scope("train_model"):
            embeddings = tf.Variable(
                tf.random_uniform(shape=(self.vocabulary_size, self.embed_size), minval=-1, maxval=1),
                name="embeddings")
            saver = tf.train.Saver([embeddings])

            embed = tf.nn.embedding_lookup(embeddings, train_batch)
            nce_weight = tf.Variable(
                tf.truncated_normal(shape=(self.vocabulary_size, self.embed_size)))
            nce_bias = tf.Variable(
                tf.zeros(shape=(self.vocabulary_size)))

        with tf.name_scope("train"):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weight,
                    biases=nce_bias,
                    labels=label_batch,
                    inputs=embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocabulary_size))

            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        with tf.name_scope("init_vars"):
            init = tf.global_variables_initializer()

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess, coord)

            writer = tf.summary.FileWriter("./CheckPoint", sess.graph)
            writer.add_graph(sess.graph)
            sess.run(init)
            for i in range(100):
                _, _loss = sess.run([optimizer, loss])
                print(i, " ", _loss)

            saver.save(sess, "./CheckPoint/embedding_set")
            coord.request_stop()
            coord.join(thread)

    def wordvec_map(self):
        tf.reset_default_graph()
        embeddings = tf.Variable(
            tf.zeros(shape=(self.vocabulary_size, self.embed_size)),
            name="embeddings")

        words = self.word_count("./DataSet/songdata.txt")

        saver = tf.train.Saver([embeddings])

        with tf.Session() as sess:
            saver.restore(sess, "./CheckPoint/embedding_set")
            _embeddings = sess.run(embeddings)
        
        print("word2vec mapping in progress...")
        wordvec_map = {"UNK":_embeddings[0].tolist()}
        for w in words:
            if len(wordvec_map) < self.vocabulary_size:
                wordvec_map[w[0]] = _embeddings[len(wordvec_map)].tolist()

        print("saving json...")
        
        with open('./DataSet/word2vec_map.json', 'w') as outfile:
            json.dump(wordvec_map, outfile)
            
if __name__ == "__main__":
    with open('./DataSet/word2vec_map.json') as data_file:    
        data = json.load(data_file)

    print(data["the"])
    print(type(data["the"]))
    print(type(data["the"][0]))
