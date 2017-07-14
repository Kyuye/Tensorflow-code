
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor

import collections
import numpy as np
import pandas
import json
import csv
import os
from pprint import pprint

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("vocabulary_size", 50000, "vocabulary size")
tf.flags.DEFINE_integer("max_document_length", 150, "max document(sentence) length")
tf.flags.DEFINE_string("train_data", "/dataset/twitter_emotion_v2(p,n,N).csv", "train data path")
tf.flags.DEFINE_string("word_vec_map_file", '/dataset/word2vec_map.json', "mapfile for word2vec")
tf.flags.DEFINE_integer("batch_size", 50, "batch size for training")
tf.flags.DEFINE_integer("regularizer_scale", 0.9, "reguarizer scale")
tf.flags.DEFINE_integer("embed_dim", 300, "embedding dimension")
tf.flags.DEFINE_integer("g_hidden1", 50, "g function 1st hidden layer unit")
tf.flags.DEFINE_integer("g_hidden2", 50, "g function 1st hidden layer unit")
tf.flags.DEFINE_integer("g_hidden3", 50, "g function 1st hidden layer unit")
tf.flags.DEFINE_integer("g_logits", 50, "g function logits")
tf.flags.DEFINE_integer("f_hidden1", 50, "f function 1st hidden layer unit")
tf.flags.DEFINE_integer("f_hidden2", 50, "f function 2nd hidden layer unit")
tf.flags.DEFINE_integer("f_logits", 50, "f function logits")
tf.flags.DEFINE_integer("emotion_class", 3, "number of emotion classes")
tf.flags.DEFINE_integer("memory_size", 10, "LSTM cell(memory) size")
tf.flags.DEFINE_string("log_dir", "./logs/", "path to logs directory")



class WasserstienGAN(object):
    def __init__(self, clip_values=(-0.01, 0.01), critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.clip_values = clip_values
        self.max_document_length = FLAGS.max_document_length
<<<<<<< HEAD
        self.max_object_pairs_num = 11174 #150C2
=======
        self.max_object_pairs_num = 11174 # 150C2
>>>>>>> 2180b3510f9bb6240f6d9aeb4280f839a1d6f890
        print("reading data..")
        self.data = self.read_datafile(os.getcwd() + FLAGS.train_data)
        print("file loaded size :", len(self.data))
        self.train_batch = tf.placeholder(tf.float32, [FLAGS.batch_size, 150, 300])
        self.label_indices = tf.placeholder(tf.int32, [FLAGS.batch_size,])
        print("reading map...")
        self.embedding_map = self.load_embedding_map(os.getcwd() + FLAGS.word_vec_map_file)
        print("session opening...")
        self.open_session()


    def get_batch(self, data, iterator):
        fullbatch_size = len(data)
        train_batch = data[(iterator*FLAGS.batch_size)%fullbatch_size:((iterator+1)*FLAGS.batch_size)%fullbatch_size]
        
        seq_vec = []
        for seq in train_batch['content']:
            embeds = np.array([self.embedding_map[i] if i in self.embedding_map else self.embedding_map["UNK"] for i in seq.split(' ')])
            seq_vec.append(np.pad(embeds, ((0, 150-embeds.shape[0]), (0,0)), 'constant'))

        indices = []
        for s in train_batch['Sentiment']:
            if s == "Neg":
                indices.append(0)
            elif s == "neutral":
                indices.append(1)
            elif s == "Pos":
                indices.append(2)
            else:
                indices.append(1)
        
        return seq_vec, indices


    def load_embedding_map(self, mapfile):
        with open(mapfile) as data_file:    
            data = json.load(data_file)
        return data


    def pairing(self, embeds):
        pair_set = []
        for seq in tf.unstack(embeds):
            comb = [i for i in range(FLAGS.max_document_length)]
            ids = []
            while len(comb) > 2:
                ids += list(map(lambda x: [comb[0], x], comb))[1:]
                del comb[0]
            
            pair = tf.nn.embedding_lookup(seq, ids)
            pair_set.append(tf.concat([pair[:,0], pair[:,1]], axis=1))
            
        return tf.stack(pair_set)


    def read_datafile(self, filename):
        data = pandas.read_csv(filename, usecols=["Sentiment", "content"], nrows=100)
        data = data[data["content"] != "0"]"0"]
        data["content"] = data["content"].astype("str")
        return data


    def _generator(self, x):
        with tf.variable_scope("generator") as scope:
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.memory_size)
            out, _ = tf.contrib.rnn.static_rnn(cell=rnn_cell, inputs=x, dtype=tf.float32)
            Wo = tf.Variable(tf.truncated_normal(shape=(len(x), FLAGS.memory_size, FLAGS.embed_dim)))
            bo = tf.Variable(tf.zeros(shape=(len(x), FLAGS.embed_dim)))
            return tf.transpose(tf.matmul(out, Wo) + bo, 

    def _create_generator(self, samples):
        self.z = tf.unstack(
            tf.random_uniform(
                shape=(samples, self.max_document_length, FLAGS.embed_dim), 
                minval=-1, 
                maxval=1, 
                dtype=tf.float32), 
                axis=1)
        self.gen_data = self._generator(self.z)


    def _discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            gs = []
            g_reuse = reuse
            f_reuse = reuse
            # for i in range(self.max_object_pairs_num):
            for i in range(100):
                if i % 10 == 0:
                    print("discriminator g func processing... : ", i, "/", self.max_object_pairs_num)
                g_in = tf.reshape(x[:,i,:], (-1, 2*FLAGS.embed_dim))
                g_layer1 = tf.layers.dense(
                    inputs=g_in,
                    units=FLAGS.g_hidden1,
                    activation=tf.nn.relu,
                    use_bias=True,
                    kernel_initializer=tf.truncated_normal_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    trainable=True,
                    name="g_layer1",
                    reuse=g_reuse
                )

                g_layer2 = tf.layers.dense(
                    inputs=g_layer1,
                    units=FLAGS.g_hidden2,
                    activation=tf.nn.relu,
                    use_bias=True,
                    kernel_initializer=tf.truncated_normal_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    trainable=True,
                    reuse=g_reuse,
                    name="g_layer2"
                )

                g_layer3 = tf.layers.dense(
                    inputs=g_layer2,
                    units=FLAGS.g_hidden3,
                    activation=tf.nn.relu,
                    use_bias=True,
                    kernel_initializer=tf.truncated_normal_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    trainable=True,
                    reuse=g_reuse,
                    name="g_layer3"
                )

                g_out = tf.layers.dense(
                    inputs=g_layer3,
                    units=FLAGS.g_logits,
                    activation=tf.nn.relu,
                    use_bias=True,
                    kernel_initializer=tf.truncated_normal_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                    trainable=True,
                    reuse=g_reuse,
                    name="g_out"
                )

                gs.append(g_out)
                g_reuse=True

            print("discriminator g summation...")
            g_batch = tf.reduce_sum(tf.transpose(tf.stack(gs), [1, 0, 2]), axis=1)

            print("discriminator f func in progress...")
            f_layer1 = tf.layers.dense(
                inputs=g_batch,
                units=FLAGS.f_hidden1,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                trainable=True,
                reuse=f_reuse,
                name="f_layer1"
            )

            print("discriminator f func in progress... layer1 complete")

            f_layer2 = tf.layers.dense(
                inputs=f_layer1,
                units=FLAGS.f_hidden2,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                trainable=True,
                reuse=f_reuse,
                name="f_layer2"
            )

            print("discriminator f func in progress... layer2 complete")

            logits = tf.layers.dense(
                inputs=f_layer2,
                units=FLAGS.f_logits,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                trainable=True,
                reuse=f_reuse,
                name="f_out"
            )

            print("discriminator f func in progress... logits complete")
        
            supervised_logits = tf.layers.dense(
                inputs=logits, 
                units=FLAGS.emotion_class,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
                trainable=True,
                reuse=f_reuse,
                name="supervised_layer"
                )

            print("discriminator f func in progress... supervised logits complete")

        return logits, supervised_logits


    def _gan_loss(self, logits_real, logits_fake, supervised_logits, label, use_features=False):
        supervised_loss = tf.losses.softmax_cross_entropy(label, supervised_logits)
        discriminator_loss = tf.reduce_mean(logits_real - logits_fake) + supervised_loss
        gen_loss = tf.reduce_mean(logits_fake)
        return discriminator_loss, gen_loss


    def _create_network(self, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
        print("Setting up model...")

        print("real data pairing..")
        real_pairs = self.pairing(self.train_batch)

        print("create generator...")
        self._create_generator(FLAGS.batch_size)
        print("fake data pairing..")
        fake_pairs = self.pairing(self.gen_data)

        print("building discriminator for real data...")
        logits_real, logits_supervised = self._discriminator(real_pairs, reuse=False)
        print("building discriminator for fake data...")
        logits_fake, _ = self._discriminator(fake_pairs, reuse=True)

        # Loss calculation
        print("building gan loss graph...")
        labels = tf.one_hot(indices=self.label_indices, depth=3, on_value=1.0, off_value=0.0)
        self.discriminator_loss, self.gen_loss = self._gan_loss(logits_real, logits_fake, logits_supervised, labels)

        print("variables scoping...")
        train_variables = tf.trainable_variables()

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]

        # print(list(map(lambda x: x.op.name, self.generator_variables)))
        # print(list(map(lambda x: x.op.name, self.discriminator_variables)))

        self.saver = tf.train.Saver(self.generator_variables)

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

        print("building train op")
        self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)



    def _get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param)
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    def _train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads)


    def train_model(self, max_iterations):
        print("Training Wasserstein GAN model...")
        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
                                        var in self.discriminator_variables]

        print("variables initializing")
        self.sess.run(tf.global_variables_initializer())

        for itr in range(1, max_iterations):
            train_data, indices = self.get_batch(self.data, itr-1)
            feed_dict = {self.train_batch: train_data, self.label_indices: indices}

            print("iterations: ", itr)

            if itr < 25 or itr % 500 == 0:
                critic_itrs = 25
            else:
                critic_itrs = self.critic_iterations

            for critic_itr in range(critic_itrs):
                print("discriminator critic: ", critic_itr)
                self.sess.run(self.discriminator_train_op, feed_dict)
                self.sess.run(clip_discriminator_var_op, feed_dict)
            
            print("generator update")
            self.sess.run(self.generator_train_op, feed_dict)

            if itr % 200 == 0:
                g_loss_val, d_loss_val = self.sess.run([self.gen_loss, self.discriminator_loss])
                self.saver.save(self.sess, "./CheckPoint/rnn_GAN")
                print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))

            self.sess.close()


    def open_session(self):
        self.sess = tf.Session()
        print("train ready")


def main(argv=None):
    gan = WasserstienGAN(critic_iterations=5)
    gan._create_network()                
    gan.train_model(100)

if __name__ == "__main__":
    os.system("mkdir dataset")
    os.system("gsutil -m cp -r gs://wgan/dataset/* $(pwd)/dataset/")
    print("data set copy")
    tf.app.run()
