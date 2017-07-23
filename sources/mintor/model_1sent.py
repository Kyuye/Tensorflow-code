    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("vocabulary_size", 10000, "vocabulary size")
tf.flags.DEFINE_integer("max_document_length", 150, "max document(sentence) length")
tf.flags.DEFINE_string("train_data", "/dataset/twitter_emotion_v2(p,n,N).csv", "train data path")
tf.flags.DEFINE_string("word_vec_map_file", '/dataset/word2vec_map.json', "mapfile for word2vec")
tf.flags.DEFINE_string("log_dir", "./logs/", "path to logs directory")
tf.flags.DEFINE_string("bucket", 'jejucamp2017', "bucket name")
tf.flags.DEFINE_integer("batch_size", 32, "batch size for training")
tf.flags.DEFINE_integer("regularizer_scale", 0.9, "reguarizer scale")
tf.flags.DEFINE_integer("embed_dim", 300, "embedding dimension")
tf.flags.DEFINE_integer("g_hidden1", 256, "g function 1st hidden layer unit")
tf.flags.DEFINE_integer("g_hidden2", 256, "g function 2nd hidden layer unit")
tf.flags.DEFINE_integer("g_hidden3", 256, "g function 3rd hidden layer unit")
tf.flags.DEFINE_integer("g_logits", 256, "g function logits")
tf.flags.DEFINE_integer("f_hidden1", 256, "f function 1st hidden layer unit")
tf.flags.DEFINE_integer("f_hidden2", 512, "f function 2nd hidden layer unit")
tf.flags.DEFINE_integer("f_logits", 159, "f function logits")
tf.flags.DEFINE_integer("emotion_class", 3, "number of emotion classes")
tf.flags.DEFINE_integer("memory_size", 128, "LSTM cell(memory) size")
tf.flags.DEFINE_bool("on_cloud", False, "run on cloud or local")
tf.flags.DEFINE_integer("gpu_num", 8, "the number of GPUs")
tf.flags.DEFINE_integer("epoch", 10, "train epoch")
tf.flags.DEFINE_integer("log_step", 50, "log step")


print("vocabulary_size: ",FLAGS.vocabulary_size)
print("max_document_length: ", FLAGS.max_document_length)
print("batch_size: ", FLAGS.batch_size)
print("regularizer_scale: ",FLAGS.regularizer_scale)
print("embed_dim: ", FLAGS.embed_dim )
print("g_hidden1: ", FLAGS.g_hidden1)
print("g_hidden2: ", FLAGS.g_hidden2)
print("g_hidden3: ", FLAGS.g_hidden3)
print("g_logits: ", FLAGS.g_logits)
print("f_hidden1: ", FLAGS.f_hidden1)
print("f_hidden2: ", FLAGS.f_hidden2)
print("f_logits: ", FLAGS.f_logits)
print("memory_size: ", FLAGS.memory_size)
print("train data directory:", FLAGS.train_data)
print("log directoty:", FLAGS.log_dir)
print("log_step:", FLAGS.log_step)
print("epoch:",FLAGS.epoch)

if FLAGS.on_cloud:
    from mintor.data_loader import TrainDataLoader
    from mintor.preprocessing import Preprocessor
    from mintor.utils import *
else:
    from data_loader import TrainDataLoader
    from preprocessing import Preprocessor
    from utils import *
    

class WassersteinGAN(object):
    def __init__(self, clip_values=(-0.01, 0.01), critic_iterations=5):
        # data loader:
        # load train data and load word2vec map file
        loader = TrainDataLoader(
            bucket=FLAGS.bucket,
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
        
        # pos: sent = "men always remember love because of romance only The best love is the kind that awaken the soul that makes us reach for more that plants the fire in our hearts and brings peace to our minds That's what I hope to give you forever The greatest happiness of life is the declaration that we are loved loved for myself or rather loved in hurt of myself The best and most beautiful things in this world cannot be seen or even heard but must be felt with the heart"
        # neg :sent = "My sadness has become an addiction when i am not sad i feel lost I start to panic trying to find my way back which leads me back to my original state You were rarely wishing for the end of pain the monster said your own pain end to how it you It is the most human wish of all everyone in life is gonna hurt you you just have to figure out which people are worth the pain The World is mad and the people are sad The saddest thing is when you are feeling real down you look around and realize that there is no shoulder for you I guess that is what saying goodbye is always like jumping off an edge The worst part is making the choice to do it Once you are in the air there is nothing you can do but let go"
        # neutral 
        sent = "You cannot visit the past but thanks to modern photography you can try to create it Just ask I was a student at a school and picture her travel across returned to the site exactly 30 years later The picture decided to create some of her favorite picture from back in the day I thought it would be a fun picture project for my YouTube channel tells I was amazed at how little these places had changed Before she left he finish out her old photo albums and scan favorite images Once in she successful track down the exact locations and follow her pose from 30 years previous creating new versions of her favorite she has showed the then and now picture on her YouTube"
        self.data = [sent for _ in range(FLAGS.batch_size*FLAGS.gpu_num)]
        self.vec2word = loader.vec2word
        self.get_batch = preproc.get_batch
        self.embedding_padding = preproc.embedding_padding
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

            supervised_logits = dense_layer(
                inputs=logits, 
                units=FLAGS.emotion_class, 
                activation=None, 
                reuse=reuse, 
                name="supervised_layer")

        return logits, supervised_logits


    def _gan_loss(self, logits_real, logits_fake, supervised_real, supervised_fake, use_features=False):
        supervised_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(supervised_real), supervised_real)
        supervised_fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(supervised_fake), supervised_fake)
        supervised_loss = supervised_real_loss + supervised_fake_loss
        # supervised loss (RN) + WGAN loss 
        discriminator_loss = tf.reduce_mean(logits_real - logits_fake) + supervised_loss
        
        gen_loss = tf.reduce_mean(logits_fake)
        tf.summary.scalar('discriminator_loss', discriminator_loss)
        tf.summary.scalar('gen_loss', gen_loss)
        tf.summary.scalar('supervised_loss', supervised_loss)
        
        return discriminator_loss, gen_loss


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
                logits_real, supervised_real = self._discriminator(real_pairs, reuse)
                logits_fake, supervised_fake = self._discriminator(fake_pairs, True)

                print("GPU:%d   building gan loss ..."%g)
                self.disc_loss, self.gen_loss = self._gan_loss(
                    logits_real, logits_fake, supervised_real, supervised_fake)

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
        print("Training Wasserstein GAN model...")
        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
                                        var in self.disc_variables]

        print("variables initializing")
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        train_step = 40000//(FLAGS.batch_size*FLAGS.gpu_num)
        
        for epoch in range(max_iterations): 
            for itr in range(1, train_step):
                # train_data, indices = self.get_batch(self.data, itr-1)
                train_data = self.embedding_padding(self.data)
                feed_dict = {}
                for g in range(FLAGS.gpu_num):
                    feed_dict[self.train_batch[g]] = train_data[g*FLAGS.batch_size:(g+1)*FLAGS.batch_size]
                    # feed_dict[self.label_indices[g]] = indices[g*FLAGS.batch_size:(g+1)*FLAGS.batch_size]
                        

                if itr < 25 or itr % 500 == 0:
                    critic_itrs = 25
                else:
                    critic_itrs = self.critic_iterations

                for critic_itr in range(critic_itrs):
                    # print("discriminator critic: ", critic_itr)
                    self.sess.run(self.disc_train_op, feed_dict)
                    self.sess.run(clip_discriminator_var_op, feed_dict)
                
                # print("generator update")
                summary, _ = self.sess.run([merged, self.gen_train_op], feed_dict)

                if itr+itr*FLAGS.epoch % FLAGS.log_step == 0:
                    g_loss_val, d_loss_val = self.sess.run(
                        [self.gen_loss, self.disc_loss], feed_dict)
                    self.saver.save(self.sess, "gs://jejucamp2017/logs/wgan")
                    summary_writer.add_summary(summary, itr)
                    print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr+itr*FLAGS.epoch, g_loss_val, d_loss_val))


    def _get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param, name="optim")
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    def evaluation(self):
        gen_data = self.sess.run(self.gen_data)
        
        seq = ""
        for w in gen_data[0]:
            seq += self.vec2word(w) + " "
            print(seq)

        with open(e"./generated_text.txt", 'w') as f:
            f.write(seq)

        os.system("gsutil -m cp -r generated_text.txt gs://jejucamp2017/logs")

    def _open_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        print("allow_growth:",config.gpu_options.allow_growth)
        print("soft placement :" ,config.allow_soft_placement)
       
        
        
        self.sess = tf.Session(config=config)
        print("train ready")
       
        
        # OLD one
        # self.sess = tf.Session(config=tf.ConfigProto(
        # allow_soft_placement=True, log_device_placement=True))
       

        # NEW one 
        
        # config.gpu_options.allow_growth = True
        # config.allow_soft_placement = False
        # config.log_device_placement = False
        # NEW:config.gpu_options.per_process_gpu_memory_fraction = 0.4
        # self.sess = tf.Session(config=config)
        # print("train ready")


      
       
        

def main(argv=None):
    gan = WassersteinGAN(critic_iterations=5)
    gan.create_network()                
    gan.train_model(FLAGS.epoch)
    gan.evaluation()
    gan.sess.close()


if __name__ == "__main__":    
    tf.app.run()

