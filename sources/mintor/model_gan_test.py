    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("vocabulary_size", 50000, "vocabulary size")
tf.flags.DEFINE_integer("max_document_length", 150, "max document(sentence) length")
tf.flags.DEFINE_string("train_data", "/dataset/twitter_emotion_v2(p,n,N).csv", "train data path")
tf.flags.DEFINE_string("word_vec_map_file", '/dataset/word2vec_map.json', "mapfile for word2vec")
tf.flags.DEFINE_integer("batch_size", 32, "batch size for training")
tf.flags.DEFINE_integer("regularizer_scale", 0.9, "reguarizer scale")
tf.flags.DEFINE_integer("embed_dim", 300, "embedding dimension")
tf.flags.DEFINE_integer("g_hidden1", 256, "g function 1st hidden layer unit")
tf.flags.DEFINE_integer("g_hidden2", 256, "g function 2nd hidden layer unit")
tf.flags.DEFINE_integer("g_hidden3", 256, "g function 3rd hidden layer unit")
tf.flags.DEFINE_integer("g_logits", 256, "g function logits")
tf.flags.DEFINE_integer("f_hidden1", 256, "f function 1st hidden layer unit")
tf.flags.DEFINE_integer("f_hidden2", 512, "f function 2nd hidden layer unit")
tf.flags.DEFINE_integer("f_logits", 1, "f function logits")
tf.flags.DEFINE_integer("emotion_class", 3, "number of emotion classes")
<<<<<<< HEAD
tf.flags.DEFINE_integer("memory_size", 128, "LSTM cell(memory) size")
tf.flags.DEFINE_string("log_dir", "gs://jejucamp2017/logs/", "path to logs directory")
tf.flags.DEFINE_bool("on_cloud", True, "run on cloud or local")
tf.flags.DEFINE_integer("gpu_num", 1, "the number of GPUs")
tf.flags.DEFINE_integer("train_step", 1000, "the train step" )
tf.flags.DEFINE_integer("log_step", 500, "the log step")
=======
tf.flags.DEFINE_integer("memory_size", 32, "LSTM cell(memory) size")
tf.flags.DEFINE_string("log_dir", "gs://wgan/logs/", "path to logs directory")
tf.flags.DEFINE_bool("on_cloud", False, "run on cloud or local")
tf.flags.DEFINE_integer("gpu_num", 1, "the number of GPUs")
tf.flags.DEFINE_integer("train_step", 100, "the train step" )
tf.flags.DEFINE_integer("log_step", 1, "the log step")
>>>>>>> 5bfe3addfb38177e6fa3e201d3e56b9a58c22ff6

print("vocabulary_size: ", FLAGS.vocabulary_size)
print("max_document_length: ", FLAGS.max_document_length)
print("batch_size: ", FLAGS.batch_size)
print("regularizer_scale: ", FLAGS.regularizer_scale)
print("embed_dim: ", FLAGS.embed_dim)
print("g_hidden1: ", FLAGS.g_hidden1)
print("g_hidden2: ", FLAGS.g_hidden2)
print("g_hidden3: ", FLAGS.g_hidden3)
print("g_logits: ", FLAGS.g_logits)
print("f_hidden1: ", FLAGS.f_hidden1)
print("f_hidden2: ", FLAGS.f_hidden2)
print("f_logits: ", FLAGS.f_logits)
print("memory_size: ", FLAGS.memory_size)
print("train_step: ", FLAGS.train_step)
print("log_step: ", FLAGS.log_step)
print("train data directory:", FLAGS.train_data)
print("log directoty:", FLAGS.Log_dir)

if FLAGS.on_cloud:
    from mintor.data_loader import TrainDataLoader
    from mintor.preprocessing import Preprocessor
    from mintor.utils import *
else:
    from data_loader import TrainDataLoader
    from preprocessing import Preprocessor
    from utils import *
    

class GAN(object):
    def __init__(self, clip_values=(-0.01, 0.01), critic_iterations=5, is_train=True):
        # data loader:
        # load train data and load word2vec map file
        if is_train:
            loader = TrainDataLoader(
                train_data_csv=FLAGS.train_data, 
                word2vec_map_json=FLAGS.word_vec_map_file, 
                on_cloud=FLAGS.on_cloud)

            # preprocessor:
            # get batch and pairing 
            preproc = Preprocessor(
                embedding_map=loader.embedding_map, 
                batch_size=FLAGS.batch_size*FLAGS.gpu_num, 
                max_document_length=FLAGS.max_document_length)

            self.critic_iterations = critic_iterations
            self.clip_values = clip_values
            self.max_object_pairs_num = preproc.max_object_pairs_num
            self.data = loader.train_data
            self.vec2word = loader.vec2word

            self.get_batch = preproc.get_batch
            self.pairing = preproc.pairing

        print("session opening...")
        self._open_session()


    def _generator(self, reuse=False):
        z = rand((FLAGS.batch_size, FLAGS.max_document_length, FLAGS.embed_dim))
        time_step = len(z)

        with tf.variable_scope('generator', reuse=reuse) as scope:
            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=FLAGS.memory_size)
            out, _ = tf.contrib.rnn.static_rnn(
                cell=rnn_cell, inputs=z, dtype=tf.float32, scope=scope)

            Wo = LSTM_Wo(shape=(time_step, FLAGS.memory_size, FLAGS.embed_dim), reuse=reuse)
            bo = LSTM_bo(shape=(time_step, FLAGS.embed_dim), reuse=reuse)

            logits = [tf.matmul(out[i], Wo[i]) + bo[i] for i in range(time_step)]

        # transpose shape to (batch, time_step, vector)
        return tf.transpose(tf.stack(logits), [1, 0, 2])


    def _discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            g_input_shape = (FLAGS.batch_size*self.max_object_pairs_num, 2*FLAGS.embed_dim)
            g_output_shape = (FLAGS.batch_size, self.max_object_pairs_num, FLAGS.g_logits)

            g_in = tf.reshape(tensor=x, shape=g_input_shape)
            
            g_layer1 = dense_layer(
                inputs=g_in, units=FLAGS.g_hidden1, reuse=reuse, name="g_layer1")
            
            g_layer2 = dense_layer(
                inputs=g_layer1, units=FLAGS.g_hidden2, reuse=reuse, name="g_layer2")
            
            g_layer3 = dense_layer(
                inputs=g_layer2, units=FLAGS.g_hidden3, reuse=reuse, name="g_layer3")
                        
            g_out = dense_layer(
                inputs=g_layer3, units=FLAGS.g_logits, reuse=reuse, name="g_out")

            g_out = tf.reshape(tensor=g_out, shape=(g_output_shape))
            g_sum = tf.reduce_sum(g_out, axis=1)

            f_layer1 = dense_layer(
                inputs=g_sum,units=FLAGS.f_hidden1, reuse=reuse, name="f_layer1")
            
            f_layer2 = dense_layer(
                inputs=f_layer1, units=FLAGS.f_hidden2, reuse=reuse, name="f_layer2")
            
            logits = dense_layer(
                inputs=f_layer2, units=FLAGS.f_logits, reuse=reuse, name="f_out")

        return logits


    def _gan_loss(self, logits_real, logits_fake, use_features=False):
        disc_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real)
        disc_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits_fake)
        disc_loss = disc_loss_real + disc_loss_fake

        gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_fake),logits_fake)
        return disc_loss, gen_loss, 


    def create_network(self, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
        print("Setting up model...")

        self.train_batch = []
        self.label_indices = []

        for g in range(FLAGS.gpu_num):
            with tf.device("/gpu:%d"%g):
                
                reuse = g > 0
                
                self.train_batch.append(tf.placeholder(
                    dtype=tf.float32, 
                    shape=[FLAGS.batch_size, FLAGS.max_document_length, FLAGS.embed_dim]))
        
                self.label_indices.append(tf.placeholder(
                    dtype=tf.int32, 
                    shape=[FLAGS.batch_size,]))

                print("GPU:%d   object pairing.."%g)
                self.gen_data = self._generator(reuse)
                fake_pairs = self.pairing(self.gen_data)
                real_pairs = self.pairing(self.train_batch[g])

                print("GPU:%d   building discriminator"%g)
                logits_real = self._discriminator(real_pairs, reuse)
                logits_fake = self._discriminator(fake_pairs, True)

                self.prob_real = tf.nn.sigmoid(logits_real)
                self.prob_fake = tf.nn.sigmoid(logits_fake)

                tf.summary.scalar("prob_real", self.prob_real)
                tf.summary.scalar("prob_fake", self.prob_fake)
                
                print("GPU:%d   building gan loss ..."%g)
                # labels = one_hot(self.label_indices[g])
                self.disc_loss, self.gen_loss = self._gan_loss(logits_real, logits_fake)

                tf.summary.scalar("discriminator_loss", self.disc_loss)
                tf.summary.scalar("generator_loss", self.gen_loss)

                print("GPU:%d   variables scoping..."%g)
                train_variables = tf.trainable_variables()

                self.gen_variables = [v for v in train_variables if v.name.startswith("generator")]
                self.disc_variables = [v for v in train_variables if v.name.startswith("discriminator")]

                # print(list(map(lambda x: x.op.name, self.gen_variables)))
                # print(list(map(lambda x: x.op.name, self.disc_variables)))

                print("GPU:%d   gradient computing ..."%g)
                self.optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

                if g == 0:
                    self.gen_grads = self.optim.compute_gradients(self.gen_loss, self.gen_variables)
                    self.disc_grads = self.optim.compute_gradients(self.disc_loss, self.disc_variables)
                else:
                    self.gen_grads += self.optim.compute_gradients(self.gen_loss, self.gen_variables)
                    self.disc_grads += self.optim.compute_gradients(self.disc_loss, self.disc_variables)

        print("build train op")
        self.gen_train_op = self.optim.apply_gradients(self.gen_grads)
        self.disc_train_op = self.optim.apply_gradients(self.disc_grads)

        self.saver = tf.train.Saver(self.gen_variables)

    def train_model(self, max_iterations):
        print("Training GAN model...")

        print("variables initializing")
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        for itr in range(1, max_iterations):
            train_data, indices = self.get_batch(self.data, itr-1)
            feed_dict = {}
            for g in range(FLAGS.gpu_num):
                feed_dict[self.train_batch[g]] = train_data[g*FLAGS.batch_size:(g+1)*FLAGS.batch_size]

            self.sess.run(self.disc_train_op, feed_dict)
            summary, _ = self.sess.run([merged, self.gen_train_op], feed_dict)

            if itr % FLAGS.log_step == 0:
                prob_real, prob_fake = self.sess.run(
                    [self.prob_real, self.prob_fake], feed_dict)
                self.saver.save(self.sess, FLAGS.log_dir+"wgan")
                summary_writer.add_summary(summary, itr)
                print("Step: %d, prob real: %g, prob fake: %g" % (itr, prob_real, prob_fake))


    def _get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param, name="optim")
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    # def evaluation(self):
    #     gen_data = self.sess.run(self.gen_data)
        
    #     seq = ""
    #     for w in gen_data[0]:
    #         seq += self.vec2word(w) + " "
            
    #     with open("./generated_text.txt", 'w') as f:
    #         f.write(seq)

    #     os.system("gsutil -m cp -r generated_text.txt gs://wgan/logs")

    def evaluation(self):
        os.system("gsutil -m cp -r gs://wgan/logs/wgan.* $(pwd)/sources/logs/")
        os.system("gsutil -m cp -r gs://wgan/logs/checkpoint $(pwd)/sources/logs/")
        
        self._generator()
        saver = tf.train.Saver()

        saver.restore(self.sess, "./sources/logs/wgan")
        gen_data = self.sess.run(self.gen_data)
        seq = ""
        for w in gen_data[0]:
            print(w)
            seq+=self.vec2word(w) + " "

<<<<<<< HEAD
        os.system("gsutil -m cp -r generated_text.txt gs://jejucamp2017/logs")
=======
>>>>>>> 5bfe3addfb38177e6fa3e201d3e56b9a58c22ff6

    def _open_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)
        print("allow_growth: ", config.gpu_options.allow_growth)
        print("soft placement: ", config.allow_soft_placement)
        print("train ready")


def main(argv=None):
<<<<<<< HEAD
    gan = GAN(critic_iterations=5)
=======
    gan = GAN(is_train=True)
>>>>>>> 5bfe3addfb38177e6fa3e201d3e56b9a58c22ff6
    gan.create_network()                
    gan.train_model(FLAGS.train_step)
    # gan.evaluation()
    gan.sess.close()


if __name__ == "__main__":    
    tf.app.run()

