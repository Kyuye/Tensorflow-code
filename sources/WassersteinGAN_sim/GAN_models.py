
import tensorflow as tf
import numpy as np
import os, sys, inspect
import time
import utils as utils

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

class GAN(object):
    def __init__(self, z_dim=0, crop_image_size=0, resized_image_size=0, batch_size=0, data_dir=0):
        self.sim_data = tf.constant([ list(range(10)) for i in range(1000)], tf.float32)
        

    def _generator(self, x):
        with tf.variable_scope("generator") as scope:
            layer1 = tf.layers.dense(x, 10, tf.nn.relu, True, tf.truncated_normal_initializer)
            layer2 = tf.layers.dense(layer1, 10, tf.nn.relu, True, tf.truncated_normal_initializer)
            self.prediction = tf.layers.dense(layer2, 10, None, True, tf.truncated_normal_initializer)
        return self.prediction


    def _discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            layer1 = tf.layers.dense(x, 10, lambda x: tf.maximum(0.2 * x, x), True, tf.truncated_normal_initializer, name="layer1", reuse=reuse)
            layer2 = tf.layers.dense(layer1, 10, lambda x: tf.maximum(0.2 * x, x), True, tf.truncated_normal_initializer, name="layer2", reuse=reuse)
            prediction = tf.layers.dense(layer2, 1, lambda x: tf.maximum(0.2 * x, x), True, tf.truncated_normal_initializer, name="prediction", reuse=reuse)
            return tf.nn.sigmoid(prediction), prediction, prediction
        

    def _cross_entropy_loss(self, prediction, labels, name="x_entropy"):
        return tf.losses.sigmoid_cross_entropy(labels, prediction)

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
        for grad, var in grads:
            return optimizer.apply_gradients(grads)

    # def _setup_placeholder(self):
    #     self.train_phase = tf.placeholder(tf.bool)
    #     self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="z")

    def _gan_loss(self, logits_real, logits_fake, feature_real, feature_fake):
        disc_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real)
        disc_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits_fake)
        self.discriminator_loss = disc_loss_real + disc_loss_fake

        gen_loss_disc = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_fake), logits_fake)
        gen_loss_features = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (10 ** 2)
        self.gen_loss = gen_loss_disc + 0.1 * gen_loss_features


    def create_network(self, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
        print("Setting up model...")
        self.z = tf.random_uniform((1000, 10), -1, 1, tf.float32)
        self.gen_data = self._generator(self.z)

        self.discriminator_real_prob, logits_real, feature_real = self._discriminator(self.sim_data, reuse=False)
        self.discriminator_fake_prob, logits_fake, feature_fake = self._discriminator(self.gen_data, reuse=True)

        # Loss calculation
        self._gan_loss(logits_real, logits_fake, feature_real, feature_fake)

        train_variables = tf.trainable_variables()

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        print(list(map(lambda x: x.op.name, self.generator_variables)))
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        print(list(map(lambda x: x.op.name, self.discriminator_variables)))

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)

    def initialize_network(self, logs_dir="./"):
        print("Initializing network...")
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def train_model(self, max_iterations):
        print("Training model...")
        for itr in range(1, max_iterations):
            self.sess.run(self.discriminator_train_op)
            self.sess.run(self.generator_train_op)

            if itr % 100 == 0:
                g_loss_val, d_loss_val, _real, _fake = self.sess.run([self.gen_loss, self.discriminator_loss, self.discriminator_real_prob, self.discriminator_fake_prob])
                print("Step: %d, generator loss: %g, discriminator_loss: %g, real: %g, fake: %g" % (itr, g_loss_val, d_loss_val, _real[0], _fake[0]))
                # self.summary_writer.add_summary(summary_str, itr)

        _pred = self.sess.run(self.prediction)
        print(_pred[0])
        self.sess.close()


    # def visualize_model(self):
    #     print("Sampling images from model...")
    #     self.z_vec = tf.random_uniform((self.batch_size, self.z_dim), -1.0, 1.0, tf.float32)
    #     feed_dict = {self.train_phase: False}

    #     images = self.sess.run(self.gen_images, feed_dict=feed_dict)
    #     images = utils.unprocess_image(images, 127.5, 127.5).astype(np.uint8)
    #     shape = [4, self.batch_size // 4]
    #     utils.save_imshow_grid(images, self.logs_dir, "generated.png", shape=shape)


class WasserstienGAN(GAN):
    def __init__(self, z_dim, crop_image_size, resized_image_size, batch_size, data_dir, clip_values=(-0.01, 0.01),
                 critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.clip_values = clip_values
        GAN.__init__(self, z_dim, crop_image_size, resized_image_size, batch_size, data_dir)


    def _generator(self, z, dims, train_phase, activation=tf.nn.relu, scope_name="generator"):
        N = len(dims)
        # image_size = 10 // (2 ** (N - 1))
        with tf.variable_scope(scope_name) as scope:
            h_z = tf.layers.dense(z, dims[0], use_bias=False, kernel_initializer=tf.truncated_normal_initializer())
            print("z:", z)
            print("h_z:", h_z)
            h_bnz = tf.layers.batch_normalization(h_z)
            h = activation(h_bnz, name='h_z')

            h_conv = tf.layers.conv1d(h, 64, 4)
            h_bn = tf.layers.batch_normalization(h_conv)
            h = activation(h_bn, name='h_%d' % index)

            image_size *= 2
            W_pred = utils.weight_variable([4, 4, dims[-1], dims[-2]], name="W_pred")
            b = tf.zeros([dims[-1]])
            deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[-1]])
            h_conv_t = utils.conv2d_transpose_strided(h, W_pred, b, output_shape=deconv_shape)
            pred_image = tf.nn.tanh(h_conv_t, name='pred_image')

        return pred_image

    def _discriminator(self, input_images, dims, train_phase, activation=tf.nn.relu, scope_name="discriminator",
                       scope_reuse=False):
        N = len(dims)
        print(N)
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = input_images
            skip_bn = True  # First layer of discriminator skips batch norm
            for index in range(N - 2):
                W = utils.weight_variable([4, 1, dims[index], dims[index + 1]], name="W_%d" % index)
                b = tf.zeros([dims[index+1]])
                print(h)
                h_conv = utils.conv2d_strided(h, W, b)

                if skip_bn:
                    h_bn = h_conv
                    skip_bn = False
                else:
                    h_bn = utils.batch_norm(h_conv, dims[index + 1], train_phase, scope="disc_bn%d" % index)
                h = activation(h_bn, name="h_%d" % index)
                # utils.add_activation_summary(h)

            W_pred = utils.weight_variable([4, 4, dims[-2], dims[-1]], name="W_pred")
            b = tf.zeros([dims[-1]])
            h_pred = utils.conv2d_strided(h, W_pred, b)
        return None, h_pred, None  # Return the last convolution output. None values are returned to maintatin disc from other GAN

    def _gan_loss(self, logits_real, logits_fake, feature_real, feature_fake, use_features=False):
        self.discriminator_loss = tf.reduce_mean(logits_real - logits_fake)
        self.gen_loss = tf.reduce_mean(logits_fake)

        # tf.scalar_summary("Discriminator_loss", self.discriminator_loss)
        # tf.scalar_summary("Generator_loss", self.gen_loss)

    def train_model(self, max_iterations):
        try:
            print("Training Wasserstein GAN model...")
            clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
                                         var in self.discriminator_variables]

            start_time = time.time()

            def get_feed_dict(train_phase=True):
                batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                feed_dict = {self.z_vec: batch_z, self.train_phase: train_phase}
                return feed_dict

            for itr in range(1, max_iterations):
                if itr < 25 or itr % 500 == 0:
                    critic_itrs = 25
                else:
                    critic_itrs = self.critic_iterations

                for critic_itr in range(critic_itrs):
                    self.sess.run(self.discriminator_train_op, feed_dict=get_feed_dict(True))
                    self.sess.run(clip_discriminator_var_op)

                feed_dict = get_feed_dict(True)
                self.sess.run(self.generator_train_op, feed_dict=feed_dict)

                if itr % 100 == 0:
                    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, itr)

                if itr % 200 == 0:
                    stop_time = time.time()
                    duration = (stop_time - start_time) / 200.0
                    start_time = stop_time
                    g_loss_val, d_loss_val = self.sess.run([self.gen_loss, self.discriminator_loss],
                                                           feed_dict=feed_dict)
                    print("Time: %g/itr, Step: %d, generator loss: %g, discriminator_loss: %g" % (
                        duration, itr, g_loss_val, d_loss_val))

                if itr % 5000 == 0:
                    self.saver.save(self.sess, self.logs_dir + "model.ckpt", global_step=itr)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print("Ending Training...")
        finally:
            self.coord.request_stop()
            self.coord.join(self.threads)  # Wait for threads to finish.


def main(argv=None):
    # gen_dim = FLAGS.gen_dimension
    # generator_dims = [64 * gen_dim, 64 * gen_dim // 2, 64 * gen_dim // 4, 64 * gen_dim // 8, 3]
    # discriminator_dims = [1, 64, 64 * 2, 64 * 4, 64 * 8, 1]

    # crop_image_size, resized_image_size = map(int, FLAGS.image_size.split(','))
    # if FLAGS.model == 0:
    #     model = GAN(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir)
    # elif FLAGS.model == 1:
    #     model = WasserstienGAN(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir,
    #                            clip_values=(-0.01, 0.01), critic_iterations=5)
    # else:
    #     raise ValueError("Unknown model identifier - FLAGS.model=%d" % FLAGS.model)

    # model.create_network(generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate,
    #                      FLAGS.optimizer_param)

    # model.initialize_network(FLAGS.logs_dir)

    # if FLAGS.mode == "train":
    #     model.train_model(int(1 + FLAGS.iterations))
    # elif FLAGS.mode == "visualize":
    #     model.visualize_model()

    gan = GAN()
    gan.create_network()
    gan.initialize_network()
    gan.train_model(20000)



if __name__ == "__main__":
    tf.app.run()
