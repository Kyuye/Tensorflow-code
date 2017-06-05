
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
        # self.sim_data = tf.unstack(tf.constant(
        #     [[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]] 
        #     if i % 2 == 0 else 
        #     [[10], [9], [8], [7], [6], [5], [4], [3], [2], [1]] 
        #     for i in range(1000)], tf.float32), axis=1)

        self.sim_data = tf.unstack(tf.constant(
            [[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]] 
            for i in range(1000)], tf.float32), axis=1)

        self.critic_iterations = critic_iterations
        self.clip_values = clip_values


    def _generator(self, x):
        with tf.variable_scope("generator") as scope:
            rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=20)
            out, _ = tf.contrib.rnn.static_rnn(cell=rnn_cell, inputs=x, dtype=tf.float32)
            Wo = tf.Variable(tf.truncated_normal(shape=(len(x), 20, 1)))
            return tf.unstack(tf.matmul(out, Wo))


    def _discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            prediction = []
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
                prediction += [[
                    tf.layers.dense(
                        inputs=layer2, 
                        units=1, 
                        activation=lambda x: tf.maximum(0.2 * x, x), 
                        use_bias=True, 
                        kernel_initializer=tf.truncated_normal_initializer, 
                        name="prediction_s%d"%i, 
                        reuse=reuse)]]
            # Return the last convolution output. None values are returned to maintatin disc from other GAN
        return tf.reduce_mean(prediction)


    def _gan_loss(self, logits_real, logits_fake, use_features=False):
        discriminator_loss = tf.reduce_mean(logits_real - logits_fake)
        gen_loss = tf.reduce_mean(logits_fake)
        return discriminator_loss, gen_loss


    def create_network(self, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
        print("Setting up model...")
        self.z = tf.unstack(tf.random_uniform((1000, 10, 1), -1, 1, tf.float32), axis=1)
        self.gen_data = self._generator(self.z)

        logits_real = self._discriminator(self.sim_data, reuse=False)
        logits_fake = self._discriminator(self.gen_data, reuse=True)

        # Loss calculation
        self.discriminator_loss, self.gen_loss = self._gan_loss(logits_real, logits_fake)

        train_variables = tf.trainable_variables()

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        print(list(map(lambda x: x.op.name, self.generator_variables)))
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        print(list(map(lambda x: x.op.name, self.discriminator_variables)))

        self.saver = tf.train.Saver(self.generator_variables)

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)

    def initialize_network(self):
        print("Initializing network...")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

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

    def destroy(self):
        self.sess.close()

    def predict(self):
        self.saver.restore(self.sess, "./CheckPoint/rnn_GAN")
        _pred = self.sess.run(tf.transpose(tf.reduce_mean(self.gen_data, axis=2)))
        print(_pred[:10])
        with open("generator_outputs.txt", 'w') as f:
            for i in _pred:
                f.writelines(str(i.round())+"\n\n")


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
    gan.create_network(optimizer="Adam")
    gan.initialize_network()
    gan.train_model(20000)
    gan.predict()
    gan.destroy()


if __name__ == "__main__":
    tf.app.run()
