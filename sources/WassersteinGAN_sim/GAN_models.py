import tensorflow as tf
import numpy as np
import os, sys, inspect
import time
import utils as utils


class GAN(object):
    def __init__(self, z_dim, crop_image_size, resized_image_size, batch_size, data_dir):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.images = self._read_input_queue()

    def _read_input(self):
        class DataRecord(object):
            pass

        record = DataRecord()

        record.input_image = tf.constant(list(range(10)), tf.float32)
        return record

    def _read_input_queue(self):
        print("Setting up image reader...")
        read_input = self._read_input()
        num_preprocess_threads = 4
        num_examples_per_epoch = 800
        min_queue_examples = int(0.1 * num_examples_per_epoch)
        print("Shuffling")
        input_image = tf.train.batch([read_input.input_image],
                                     batch_size=self.batch_size,
                                     num_threads=num_preprocess_threads,
                                     capacity=min_queue_examples + 2 * self.batch_size
                                     )
        return input_image

    def _generator(self, z, dims, train_phase, activation=tf.nn.relu, scope_name="generator"):
        N = len(dims)
        image_size = 10 // (2 ** (N - 1))

        W_z = utils.weight_variable([self.z_dim, dims[0] * image_size * image_size], name="W_z")
        b_z = utils.bias_variable([dims[0] * image_size * image_size], name="b_z")
        h_z = tf.matmul(z, W_z) + b_z
        h_z = tf.reshape(h_z, [-1, image_size, image_size, dims[0]])
        h_bnz = utils.batch_norm(h_z, dims[0], train_phase, scope="gen_bnz")
        h = activation(h_bnz, name='h_z')

        for index in range(N - 2):
            image_size *= 2
            W = utils.weight_variable([5, 5, dims[index + 1], dims[index]], name="W_%d" % index)
            b = utils.bias_variable([dims[index + 1]], name="b_%d" % index)
            deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[index + 1]])
            h_conv_t = utils.conv2d_transpose_strided(h, W, b, output_shape=deconv_shape)
            h_bn = utils.batch_norm(h_conv_t, dims[index + 1], train_phase, scope="gen_bn%d" % index)
            h = activation(h_bn, name='h_%d' % index)

    def _discriminator(self, input_images, dims, train_phase, activation=tf.nn.relu, scope_name="discriminator",
                       scope_reuse=False):
        N = len(dims)
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = input_images
            skip_bn = True  # First layer of discriminator skips batch norm
            for index in range(N - 2):
                W = utils.weight_variable([5, 5, dims[index], dims[index + 1]], name="W_%d" % index)
                b = utils.bias_variable([dims[index + 1]], name="b_%d" % index)
                h_conv = utils.conv2d_strided(h, W, b)
                if skip_bn:
                    h_bn = h_conv
                    skip_bn = False
                else:
                    h_bn = utils.batch_norm(h_conv, dims[index + 1], train_phase, scope="disc_bn%d" % index)
                h = activation(h_bn, name="h_%d" % index)

            shape = h.get_shape().as_list()
            image_size = self.resized_image_size // (2 ** (N - 2))  # dims has input dim and output dim
            h_reshaped = tf.reshape(h, [self.batch_size, image_size * image_size * shape[3]])
            W_pred = utils.weight_variable([image_size * image_size * shape[3], dims[-1]], name="W_pred")
            b_pred = utils.bias_variable([dims[-1]], name="b_pred")
            h_pred = tf.matmul(h_reshaped, W_pred) + b_pred

        return tf.nn.sigmoid(h_pred), h_pred, h

    def _cross_entropy_loss(self, logits, labels, name="x_entropy"):
        xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels))
        # tf.scalar_summary(name, xentropy)
        return xentropy

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

    def _setup_placeholder(self):
        self.train_phase = tf.placeholder(tf.bool)
        self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="z")

    def _gan_loss(self, logits_real, logits_fake, feature_real, feature_fake, use_features=False):
        discriminator_loss_real = self._cross_entropy_loss(logits_real, tf.ones_like(logits_real),
                                                           name="disc_real_loss")

        discriminator_loss_fake = self._cross_entropy_loss(logits_fake, tf.zeros_like(logits_fake),
                                                           name="disc_fake_loss")
        self.discriminator_loss = discriminator_loss_fake + discriminator_loss_real

        gen_loss_disc = self._cross_entropy_loss(logits_fake, tf.ones_like(logits_fake), name="gen_disc_loss")
        if use_features:
            gen_loss_features = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (self.crop_image_size ** 2)
        else:
            gen_loss_features = 0
        self.gen_loss = gen_loss_disc + 0.1 * gen_loss_features


    def create_network(self, generator_dims, discriminator_dims, optimizer="Adam", learning_rate=2e-4,
                       optimizer_param=0.9, improved_gan_loss=True):
        print("Setting up model...")
        self._setup_placeholder()
        self.gen_images = self._generator(self.z_vec, generator_dims, self.train_phase, scope_name="generator")

        def leaky_relu(x, name="leaky_relu"):
            return utils.leaky_relu(x, alpha=0.2, name=name)

        discriminator_real_prob, logits_real, feature_real = self._discriminator(self.images, discriminator_dims,
                                                                                 self.train_phase,
                                                                                 activation=leaky_relu,
                                                                                 scope_name="discriminator",
                                                                                 scope_reuse=False)

        discriminator_fake_prob, logits_fake, feature_fake = self._discriminator(self.gen_images, discriminator_dims,
                                                                                 self.train_phase,
                                                                                 activation=leaky_relu,
                                                                                 scope_name="discriminator",
                                                                                 scope_reuse=True)

        # utils.add_activation_summary(tf.identity(discriminator_real_prob, name='disc_real_prob'))
        # utils.add_activation_summary(tf.identity(discriminator_fake_prob, name='disc_fake_prob'))

        # Loss calculation
        self._gan_loss(logits_real, logits_fake, feature_real, feature_fake, use_features=improved_gan_loss)

        train_variables = tf.trainable_variables()

        # for v in train_variables:
            # print (v.op.name)

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        # print(map(lambda x: x.op.name, generator_variables))
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        # print(map(lambda x: x.op.name, discriminator_variables))

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)

    def initialize_network(self, logs_dir):
        print("Initializing network...")
        self.logs_dir = logs_dir
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def train_model(self, max_iterations):
        try:
            print("Training model...")
            for itr in range(1, max_iterations):
                batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                feed_dict = {self.z_vec: batch_z, self.train_phase: True}

                self.sess.run(self.discriminator_train_op, feed_dict=feed_dict)
                self.sess.run(self.generator_train_op, feed_dict=feed_dict)

                if itr % 10 == 0:
                    g_loss_val, d_loss_val, summary_str = self.sess.run(
                        [self.gen_loss, self.discriminator_loss, self.summary_op], feed_dict=feed_dict)
                    print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))
                    self.summary_writer.add_summary(summary_str, itr)

                if itr % 2000 == 0:
                    self.saver.save(self.sess, self.logs_dir + "model.ckpt", global_step=itr)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print("Ending Training...")
        finally:
            self.coord.request_stop()
            self.coord.join(self.threads)  # Wait for threads to finish.

    def visualize_model(self):
        print("Sampling images from model...")
        self.z_vec = tf.random_uniform((self.batch_size, self.z_dim), -1.0, 1.0, tf.float32)
        feed_dict = {self.train_phase: False}

        images = self.sess.run(self.gen_images, feed_dict=feed_dict)
        images = utils.unprocess_image(images, 127.5, 127.5).astype(np.uint8)
        shape = [4, self.batch_size // 4]
        utils.save_imshow_grid(images, self.logs_dir, "generated.png", shape=shape)


class WasserstienGAN(GAN):
    def __init__(self, z_dim, crop_image_size, resized_image_size, batch_size, data_dir, clip_values=(-0.01, 0.01),
                 critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.clip_values = clip_values
        GAN.__init__(self, z_dim, crop_image_size, resized_image_size, batch_size, data_dir)

    def _generator(self, z, dims, train_phase, activation=tf.nn.relu, scope_name="generator"):
        N = len(dims)
        image_size = 10 // (2 ** (N - 1))
        with tf.variable_scope(scope_name) as scope:
            W_z = utils.weight_variable([self.z_dim, dims[0] * image_size * image_size], name="W_z")
            h_z = tf.matmul(z, W_z)
            h_z = tf.reshape(h_z, [-1, image_size, image_size, dims[0]])
            h_bnz = utils.batch_norm(h_z, dims[0], train_phase, scope="gen_bnz")
            h = activation(h_bnz, name='h_z')

            for index in range(N - 2):
                image_size *= 2
                W = utils.weight_variable([4, 4, dims[index + 1], dims[index]], name="W_%d" % index)
                b = tf.zeros([dims[index + 1]])
                deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[index + 1]])
                h_conv_t = utils.conv2d_transpose_strided(h, W, b, output_shape=deconv_shape)
                h_bn = utils.batch_norm(h_conv_t, dims[index + 1], train_phase, scope="gen_bn%d" % index)
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
        print(input_images)
        N = len(dims)
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = input_images
            skip_bn = True  # First layer of discriminator skips batch norm
            for index in range(N - 2):
                W = utils.weight_variable([4, 1, dims[index], dims[index + 1]], name="W_%d" % index)
                b = tf.zeros([dims[index+1]])
                print(W)
                print(b)
                print(h)
                h_conv = utils.conv2d_strided(h, W, b)

                if skip_bn:
                    h_bn = h_conv
                    skip_bn = False
                else:
                    h_bn = utils.batch_norm(h_conv, dims[index + 1], train_phase, scope="disc_bn%d" % index)
                h = activation(h_bn, name="h_%d" % index)
                utils.add_activation_summary(h)

            W_pred = utils.weight_variable([4, 4, dims[-2], dims[-1]], name="W_pred")
            b = tf.zeros([dims[-1]])
            h_pred = utils.conv2d_strided(h, W_pred, b)
        return None, h_pred, None  # Return the last convolution output. None values are returned to maintatin disc from other GAN

    def _gan_loss(self, logits_real, logits_fake, feature_real, feature_fake, use_features=False):
        self.discriminator_loss = tf.reduce_mean(logits_real - logits_fake)
        self.gen_loss = tf.reduce_mean(logits_fake)

        tf.scalar_summary("Discriminator_loss", self.discriminator_loss)
        tf.scalar_summary("Generator_loss", self.gen_loss)

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

            for itr in xrange(1, max_iterations):
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
