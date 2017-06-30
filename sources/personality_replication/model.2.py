
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


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("vocabulary_size", 50000, "vocabulary size")
tf.flags.DEFINE_integer("max_document_length", 150, "max document(sentence) length")
tf.flags.DEFINE_string("train_data", "./DataSet/twitter_emotion.csv", "train data path")
tf.flags.DEFINE_integer("batch_size", 10, "batch size for training")
tf.flags.DEFINE_integer("emotion_class", 3, "emotion label classes")
tf.flags.DEFINE_integer("embed_dim", 200, "embedding dimension")
# tf.flags.DEFINE_string("logs_dir", "logs/CelebA_GAN_logs/", "path to logs directory")
# tf.flags.DEFINE_string("data_dir", "../Data_zoo/CelebA_faces/", "path to dataset")
# tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
# tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
# tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for Adam optimizer / decay for RMSProp")
# tf.flags.DEFINE_float("iterations", "1e5", "No. of iterations to train model")
# tf.flags.DEFINE_string("image_size", "108,64", "Size of actual images, Size of images to be generated at.")
# tf.flags.DEFINE_integer("model", "1", "Model to train. 0 - GAN, 1 - WassersteinGAN")
# tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer to use for training")
# tf.flags.DEFINE_integer("gen_dimension", "16", "dimension of first layer in generator")
# tf.flags.DEFINE_string("mode", "train", "train / visualize model")


class WasserstienGAN(object):
    def __init__(self, clip_values=(-0.01, 0.01), critic_iterations=5):
        self.filename = FLAGS.train_data
        self.critic_iterations = critic_iterations
        self.clip_values = clip_values
        self.max_document_length = FLAGS.max_document_length
        self.data = self.read_datafile(self.filename)
        self.word_ids, self.vocabulary_size = self.word_identify(self.data)
        self.object_pairs_set = []
        self.max_object_pairs_num = 0

    def build_pair_set(self):
        for ids in self.word_ids:
            object_pairs = []
            seq = ids.tolist()
            seq_length = np.count_nonzero(ids)
            while seq_length >= 2:
                object_pairs += list(map(lambda x: (seq[0], x), seq[:seq_length]))[1:]
                del seq[0]
                seq_length = np.count_nonzero(seq)

            self.object_pairs_set.append(object_pairs)

            if self.max_object_pairs_num < len(object_pairs):
                self.max_object_pairs_num = len(object_pairs)
    
    def build_embed_batch(self):
        embed_reuse = False
        object_pairs_list = []
        for ids in self.object_pairs_set[:100]:
            object_pairs_embed = tf.contrib.layers.embed_sequence(
                ids=ids,
                vocab_size=FLAGS.vocabulary_size,
                embed_dim=FLAGS.embed_dim,
                reuse=embed_reuse,
                scope="embeddings")
            
            object_pairs_concat = tf.reshape(
                object_pairs_embed,
                shape=(-1, 2*FLAGS.embed_dim))
            
            object_pairs_list.append(tf.pad(
                tensor=object_pairs_concat,
                paddings=[
                    [0, self.max_object_pairs_num-len(ids)], 
                    [0, 0]])
            )

            embed_reuse = True

        object_pairs = tf.stack(object_pairs_list)

        self.train_batch = tf.train.batch(
            tensors=[object_pairs], 
            batch_size=FLAGS.batch_size, 
            num_threads=4, 
            enqueue_many=True)
        
        
        gs = []
        g_reuse = False
        for i in range(5):
            x = tf.reshape(self.train_batch[:,i,:], (-1, FLAGS.embed_dim))
            gs.append(tf.layers.dense(x, 10, reuse=g_reuse, name="g_function"))
            g_reuse=True

        g_batch = tf.reduce_sum(tf.transpose(tf.stack(gs), [1, 0, 2]), axis=1)
        
        f = tf.layers.dense(g_batch, 3, name="f_function")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess, coord)
            
            a, b = sess.run([f, tf.shape(f)])
            print(a)
            print(b)
            
            coord.request_stop()
            coord.join(thread)
        
    def read_datafile(self, filename):
        data = pandas.read_csv(filename, usecols=["sentiment", "content"])
        data["content"] = data["content"].astype("str")
        return data


    def word_identify(self, dataframe):
        contents = dataframe["content"].values.tolist()
        vocab_processor = VocabularyProcessor(self.max_document_length)
        word_ids = np.array(list(vocab_processor.fit_transform(contents)))
        return word_ids, np.max(word_ids)


    def text_sort(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = list(map(lambda x: (x[1], x[3]) , reader))
            del data[0]

        with open(filename[:-4] + "sorted.csv", 'w') as f:
            writer = csv.writer(f)
            for sentiment, content in data:
                writer.writerow([sentiment, content])



    def word2vec(self, word_sequence):
        return list(map(lambda x: self.embedding_map[x], word_sequence))

    def _generator(self, x):
        with tf.variable_scope("generator") as scope:
            # TODO static_rnn --> dynamic_rnn
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=20)
            out, _ = tf.contrib.rnn.static_rnn(cell=rnn_cell, inputs=x, dtype=tf.float32)
            Wo = tf.unstack(tf.Variable(tf.truncated_normal(shape=(len(x), 20, 1))))
            bo = tf.unstack(tf.Variable(tf.zeros(shape=(len(x), 1, 1))))
            return tf.matmul(out, Wo) + bo

    def _discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            # o1 = 

            g_in = tf.concat([o1, o2], axis=0)

            g_layer1 = tf.layers.dense(
                inputs=g_in,
                units=g_hidden1,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                trainable=True,
                name="g_layer1"
            )

            g_layer2 = tf.layers.dense(
                inputs=g_layer1,
                units=g_hidden2,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                trainable=True,
                name="g_layer2"
            )

            g_layer3 = tf.layers.dense(
                inputs=g_layer2,
                units=g_hidden3,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                trainable=True,
                name="g_layer3"
            )

            g_out = tf.layers.dense(
                inputs=g_layer3,
                units=g_class,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                trainable=True,
                name="g_out"
            )

            f_in = tf.reduce_sum(g_out, axis=0)

            f_layer1 = tf.layers.dense(
                inputs=f_in,
                units=f_hidden1,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                trainable=True,
                name="f_layer1"
            )

            f_layer2 = tf.layers.dense(
                inputs=f_layer1,
                units=f_hidden2,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                trainable=True,
                name="f_layer2"
            )

            logits = tf.layers.dense(
                inputs=f_layer2,
                units=logits_dim,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                trainable=True,
                name="f_out"
            )
        
            supervised_logits = tf.layers.dense(
                inputs=logits, 
                units=FLAGS.emotion_class,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                activity_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
                trainable=True,
                name="supervised_layer"
                )

        return logits, supervised_logits

    def _create_generator(self):
        self.z = tf.unstack(
            tf.random_uniform(
                shape=(10, FLAGS.max_document_length, FLAGS.embed_dim), 
                minval=-1, 
                maxval=1, 
                dtype=tf.float32), 
                axis=1)
        self.gen_data = self._generator(self.z)
        

    def _gan_loss(self, logits_real, logits_fake, supervised_logits, use_features=False):
        supervised_loss = tf.losses.softmax_cross_entropy(label, supervised_logits)
        discriminator_loss = tf.reduce_mean(logits_real - logits_fake) + supervised_loss
        gen_loss = tf.reduce_mean(logits_fake)
        return discriminator_loss, gen_loss


    def _create_network(self, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
        print("Setting up model...")
        self._create_generator()

        logits_real, logits_supervised = self._discriminator(self.sim_data, reuse=False)
        logits_fake, _ = self._discriminator(self.gen_data, reuse=True)

        # Loss calculation
        self.discriminator_loss, self.gen_loss = self._gan_loss(logits_real, logits_fake, logits_supervised)

        train_variables = tf.trainable_variables()

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]

        # print(list(map(lambda x: x.op.name, self.generator_variables)))
        # print(list(map(lambda x: x.op.name, self.discriminator_variables)))

        self.saver = tf.train.Saver(self.generator_variables)

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)

    def initialize_network(self, mode):
        if mode is 'train':
            print("building network")
            self._create_network()
            print("variables initializing")
            self.sess.run(tf.global_variables_initializer())
            print("training...")
            self.train_model(1000)
            print("session closed")
            self.sess.close()
        elif mode is 'predict':
            print("building network")
            self._create_generator()
            print("predict..")
            self.predict()
            print("session closed")
            self.sess.close()
        else:
            print("please select the mode train/predict")
            return


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

    def predict(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "./CheckPoint/rnn_GAN")
        tf.train.write_graph(self.sess.graph_def,"./CheckPoint/",'graph.pbtxt',False)
        _pred  = self.sess.run(tf.transpose(self.gen_data, perm=[1,0,2]))
        print(_pred)


    def train_model(self, max_iterations):
        print("Training Wasserstein GAN model...")
        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
                                        var in self.discriminator_variables]

        for itr in range(1, max_iterations):
            if itr < 25 or itr % 500 == 0:
                critic_itrs = 25
            else:
                critic_itrs = self.critic_iterations

            for critic_itr in range(critic_itrs):
                self.sess.run(self.discriminator_train_op)
                self.sess.run(clip_discriminator_var_op)

            self.sess.run(self.generator_train_op)

            if itr % 200 == 0:
                g_loss_val, d_loss_val = self.sess.run([self.gen_loss, self.discriminator_loss])
                self.saver.save(self.sess, "./CheckPoint/rnn_GAN")
                print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))


def main(argv=None):
    gan = WasserstienGAN(critic_iterations=5)
    # gan.initialize_network("predict")

if __name__ == "__main__":
    tf.app.run()
