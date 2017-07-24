
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import csv
import json
import collections
import matplotlib.pyplot as plt
from pprint import pprint
from utils import *
import pandas
import numpy as np
import re 
from pandas import Series, DataFrame 
import string 


class skip_gram(object):
    def __init__(self):
        self.vocabulary_size = 10000
        self.embed_size = 300
        self.num_sampled = 64
        self.batch_size = 32

    def csv_to_text(self, file_dir):
        
        data = pandas.read_csv(file_dir, usecols=["Sentiment", "content"])
        data = data[data["content"]!= "0"]
        data["content"] = data["content"].astype("str")
        sent1 = data["content"]
        sent2 = sent1.str.split(" ")

        remove_words = ["@", "http://", "--", ":", ";","&","=",">","<","$","~","#","//","^", "(", ")", ",","\\"]
        final = filter_none(sent2)
        final = filter_word(remove_words,final)
        final = filter_printable(final)
        
        # with open(file_dir, 'r') as f:
        #     reader = csv.reader(f)
        # data = list(map(lambda x: x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10], reader))
            # del data[0]
        words=[]

        for sent in final:
            for word in sent:
                words.append(word)
                
        
        with open(file_dir[:-4]+".txt", 'w') as f:
            for i in words:
                f.write(str(i)+" ")
        
       
        # words = self.words_read_text("./DataSet/twitter_emotion_v2(p,n,N).txt")
        # count = collections.Counter(words).most_common(self.vocabulary_size)
        # print(count)

    def words_read_text(self, file_dir):
        with open(file_dir, 'r')  as f:
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


    def train_model(self,step):
        tf.reset_default_graph()
        with tf.name_scope("train_set_build"):
            filename = file_dir + 'train_set.csv'
            filename_queue = tf.train.string_input_producer([filename])

            reader = tf.TextLineReader()

            key, value = reader.read(filename_queue)

            trains, labels = tf.decode_csv(value, [[0]]*2)
            train_batch, label_batch = tf.train.batch([trains, labels], self.batch_size)
            # train_batch = tf.reshape(train_batch, shape=(-1, 1))
            label_batch = tf.reshape(label_batch, shape=(-1, 1))

        embeddings = tf.Variable(
            tf.random_uniform(shape=(self.vocabulary_size, self.embed_size), minval=-1, maxval=1),
            name="embeddings")
        saver = tf.train.Saver([embeddings])

        with tf.name_scope("train_model"):
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
            for i in range(step):
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

        words = self.word_count("./DataSet/twitter_emotion_v2(p,n,N).txt")

        saver = tf.train.Saver([embeddings])

        with tf.Session() as sess:
            saver.restore(sess, "./CheckPoint/embedding_set")
            _embeddings = sess.run(embeddings)
        
        print("word2vec mapping in progress...")
        wordvec_map = {"UNK":_embeddings[0].tolist()}
        for w, c in words:
            if len(wordvec_map) < self.vocabulary_size:
                wordvec_map[w] = _embeddings[len(wordvec_map)].tolist()

        print("saving json...")
        
        with open('./DataSet/word2vec_map.json', 'w') as outfile:
            json.dump(wordvec_map, outfile)
            


file_dir = "./DataSet/"

data_name = file_dir + "twitter_emotion_v2(p,n,N).csv"

filename = "./DataSet/twitter_emotion_v2(p,n,N).csv"

textname = "./DataSet/twitter_emotion_v2(p,n,N).txt"

jsonfile = "./DataSet/word2vec_map.json"


if __name__ == "__main__":
    with open(jsonfile) as data_file :    
        final_version = json.load(data_file)
        # print(final_version.keys())
    
    sent = u"""You cannot visit the past but thanks to modern photography you can try to create it Just ask I was a student at a school and picture her travel across returned to the site exactly 30 years later The picture decided to create some of her favorite picture from back in the day I thought it would be a fun picture project for my YouTube channel tells I was amazed at how little these places had changed Before she left he finish out her old photo albums and scan favorite images Once in she successful track down the exact locations and follow her pose from 30 years previous creating new versions of her favorite she has showed the then and now picture on her YouTube""".split(" ")


    i = 0
    for word in sent:
        print(word in final_version," ",word)
        i+=1
    
    print(i)
    exit()

    w = skip_gram()
    w.csv_to_text(filename)
    words_pair, vocab_dict = w.vocab_to_dict(textname)
    
    # print("words pair : ", words_pair)
    # print("vocab_dictionary :", vocab_dict)
   
    data = w.words_read_text(textname)

    words_id = [vocab_dict[i] if i in vocab_dict else vocab_dict["UNK"] for i in data]
    
    # print("data is ", data[:10])
    # print("words id :", words_id[:10])
    batch = w.build_train_data(words_id, 2, 20)
    w.write_train_data(batch)

    # print("batch is ", batch[:10])
    
    w.train_model(10000)
    w.wordvec_map()
    
   
    with open(jsonfile) as data_file :    
        final_version = json.load(data_file)
        print(final_version.keys())
    # words_pair, vocab_dict = vocab_to_dict(filename)
    # data = words_read_text(file_dir)
    # words = [i[0] for i in words_pair]
    # words_id = [vocab_dict[i] if i in vocab_dict else vocab_dict["UTK8"] for i in data]


    # word2vec = skip_gram()
    # word2vec.train_model()
    # word2vec.wordvec_map()




# batch = build_train_data(words_id, 2, 20)
# write_train_data(batch)


    # with open('./DataSet/word2vec_map.json') as data_file:    
    #     data = json.load(data_file)

