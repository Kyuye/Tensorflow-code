
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import os


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/CelebA_GAN_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "../Data_zoo/CelebA_faces/", "path to dataset")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for Adam optimizer / decay for RMSProp")
tf.flags.DEFINE_float("iterations", "1e5", "No. of iterations to train model")
tf.flags.DEFINE_string("image_size", "108,64", "Size of actual images, Size of images to be generated at.")
tf.flags.DEFINE_integer("model", "1", "Model to train. 0 - GAN, 1 - WassersteinGAN")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer to use for training")
tf.flags.DEFINE_integer("gen_dimension", "16", "dimension of first layer in generator")
tf.flags.DEFINE_string("mode", "train", "train / visualize model")


class WasserstienGAN(object):
    def __init__(self, clip_values=(-0.01, 0.01), critic_iterations=5):
        if os.path.exists('./DataSet/word2vec_map.json'):
            print("importing json file...")
            with open('./DataSet/word2vec_map.json') as jsonfile:    
                self.embedding_map = json.load(jsonfile)
        
        self.sim_data = tf.unstack(tf.constant(
            [[[t] for t in range(10)] 
            for _ in range(100)], 
            dtype=tf.float32), 
            axis=1)
        self.critic_iterations = critic_iterations
        self.clip_values = clip_values
        self.sess = tf.Session()

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
            x = tf.unstack(x)
            logits = []
            for i in range(10):
                layer1 = tf.layers.dense(
                    inputs=x[i], 
                    units=10, 
                    activation=lambda x: tf.maximum(0.2 * x, x), 
                    use_bias=True, 
                    kernel_initializer=tf.truncated_normal_initializer, 
                    name="layer1_s%d"%i, 
                    reuse=reuse)
                layer2 = tf.layers.dense(
                    inputs=layer1, 
                    units=10, 
                    activation=lambda x: tf.maximum(0.2 * x, x), 
                    use_bias=True, 
                    kernel_initializer=tf.truncated_normal_initializer, 
                    name="layer2_s%d"%i, 
                    reuse=reuse)
                logits += [tf.layers.dense(
                    inputs=layer2, 
                    units=10, 
                    use_bias=True, 
                    kernel_initializer=tf.truncated_normal_initializer, 
                    name="logits_s%d"%i, 
                    reuse=reuse)]
        return logits

    def _create_generator(self):
        self.z = tf.unstack(tf.random_uniform((100, 10, 1), -1, 1, tf.float32), axis=1)
        self.gen_data = self._generator(self.z)
        
    def _gan_loss(self, logits_real, logits_fake, use_features=False):
        discriminator_loss = 0
        gen_loss = 0
        for t in range(10):
            discriminator_loss += tf.reduce_mean(logits_real[t] - logits_fake[t])
            gen_loss += tf.reduce_mean(logits_fake[t])
        return discriminator_loss, gen_loss


    def _create_network(self, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
        print("Setting up model...")
        self._create_generator()

        logits_real = self._discriminator(self.sim_data, reuse=False)
        logits_fake = self._discriminator(self.gen_data, reuse=True)

        # Loss calculation
        self.discriminator_loss, self.gen_loss = self._gan_loss(logits_real, logits_fake)

        train_variables = tf.trainable_variables()

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        # print(list(map(lambda x: x.op.name, self.generator_variables)))
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
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
    gan.initialize_network("predict")

if __name__ == "__main__":
    tf.app.run()
