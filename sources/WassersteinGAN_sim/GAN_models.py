
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib import legacy_seq2seq

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
        self.sim_data = tf.constant([ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if i % 2 == 0 else [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] for i in range(1000)], tf.float32)
        self.critic_iterations = critic_iterations
        self.clip_values = clip_values
        
    def _generator(self, x):
        with tf.variable_scope("generator") as scope:
            layer1 = tf.layers.dense(x, 10, tf.nn.relu, True, tf.truncated_normal_initializer)
            layer2 = tf.layers.dense(layer1, 10, tf.nn.relu, True, tf.truncated_normal_initializer)
            return tf.layers.dense(layer2, 10, None, True, tf.truncated_normal_initializer) 
            

    def _discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            layer1 = tf.layers.dense(x, 10, lambda x: tf.maximum(0.2 * x, x), True, tf.truncated_normal_initializer, name="layer1", reuse=reuse)
            layer2 = tf.layers.dense(layer1, 10, lambda x: tf.maximum(0.2 * x, x), True, tf.truncated_normal_initializer, name="layer2", reuse=reuse)
            prediction = tf.layers.dense(layer2, 1, lambda x: tf.maximum(0.2 * x, x), True, tf.truncated_normal_initializer, name="prediction", reuse=reuse)
        # Return the last convolution output. None values are returned to maintatin disc from other GAN
        return None, prediction, None
        

    def _gan_loss(self, logits_real, logits_fake, feature_real, feature_fake, use_features=False):
        discriminator_loss = tf.reduce_mean(logits_real - logits_fake)
        gen_loss = tf.reduce_mean(logits_fake)
        return discriminator_loss, gen_loss

    def create_network(self, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
        print("Setting up model...")
        self.z = tf.random_uniform((1000, 10), -1, 1, tf.float32)
        self.gen_data = self._generator(self.z)

        self.discriminator_real_prob, logits_real, feature_real = self._discriminator(self.sim_data, reuse=False)
        self.discriminator_fake_prob, logits_fake, feature_fake = self._discriminator(self.gen_data, reuse=True)

        # Loss calculation
        self.discriminator_loss, self.gen_loss = self._gan_loss(logits_real, logits_fake, feature_real, feature_fake)


        train_variables = tf.trainable_variables()

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        print(list(map(lambda x: x.op.name, self.generator_variables)))
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        print(list(map(lambda x: x.op.name, self.discriminator_variables)))

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)

    def initialize_network(self):
        print("Initializing network...")
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

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
                print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))

        _pred = self.sess.run(tf.unstack(self.gen_data))

        for i in _pred:
            print(i)
            print()

        self.sess.close()

def main(argv=None):
    gan = WasserstienGAN()
    gan.create_network(optimizer="Adam")
    gan.initialize_network()
    gan.train_model(20000)


if __name__ == "__main__":
    tf.app.run()
