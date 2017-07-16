    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("vocabulary_size", 50000, "vocabulary size")
tf.flags.DEFINE_integer("max_document_length", 150, "max document(sentence) length")
tf.flags.DEFINE_string("train_data", "/dataset/twitter_emotion_v2(p,n,N).csv", "train data path")
tf.flags.DEFINE_string("word_vec_map_file", '/dataset/word2vec_map.json', "mapfile for word2vec")
tf.flags.DEFINE_integer("batch_size", 10, "batch size for training")
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
tf.flags.DEFINE_integer("memory_size", 20, "LSTM cell(memory) size")
tf.flags.DEFINE_string("log_dir", "./logs/", "path to logs directory")
tf.flags.DEFINE_bool("on_cloud", False, "run on cloud or local")

if FLAGS.on_cloud:
    from mintor.data_loader import TrainDataLoader
    from mintor.preprocessing import Preprocessor
else:
    from data_loader import TrainDataLoader
    from preprocessing import Preprocessor
    

class WassersteinGAN(object):
    def __init__(self, clip_values=(-0.01, 0.01), critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.clip_values = clip_values

        loader = TrainDataLoader(
            train_data_csv=FLAGS.train_data, 
            word2vec_map_json=FLAGS.word_vec_map_file, 
            on_cloud=FLAGS.on_cloud)
        
        self.preproc = Preprocessor(
            embedding_map=loader.embedding_map, 
            batch_size=FLAGS.batch_size, 
            max_document_length=FLAGS.max_document_length)

        self.data = loader.train_data

        print("session opening...")
        self.open_session()


    def _generator(self, x):
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.memory_size)
        out, _ = tf.contrib.rnn.static_rnn(cell=rnn_cell, inputs=x, dtype=tf.float32)
        Wo = tf.unstack(tf.Variable(
            tf.truncated_normal(shape=(len(x), FLAGS.memory_size, FLAGS.embed_dim))))
        bo = tf.unstack(tf.Variable(
            tf.zeros(shape=(len(x), FLAGS.embed_dim))))

        # Wo = tf.get_variable(
        #     name="Wo",
        #     shape=(len(x), FLAGS.memory_size, FLAGS.embed_dim),
        #     dtype=tf.float32,
        #     initializer=tf.truncated_normal_initializer()
        # )

        # bo = tf.get_variable(
        #     name="bo",
        #     shape=(len(x), FLAGS.embed_dim),
        #     dtype=tf.float32,
        #     initializer=tf.zeros_initializer()
        # )

        logits = []
        for i in range(len(x)):
            logits.append(tf.matmul(out[i], Wo[i]) + bo[i])

        return tf.transpose(tf.stack(logits), [1, 0, 2])

    def _create_generator(self, samples):
        self.z = tf.unstack(
            tf.random_uniform(
                shape=(samples, FLAGS.max_document_length, FLAGS.embed_dim), 
                minval=-1, 
                maxval=1, 
                dtype=tf.float32), 
                axis=1)
        self.gen_data = self._generator(self.z)


    def _discriminator(self, x):
        g_in = tf.reshape(
            tensor=x, 
            shape=(FLAGS.batch_size*self.preproc.max_object_pairs_num, 2*FLAGS.embed_dim))
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
            reuse=None
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
            reuse=None,
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
            reuse=None,
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
            reuse=None,
            name="g_out"
        )

        g_out = tf.reshape(
            tensor=g_out, 
            shape=(FLAGS.batch_size,self.preproc.max_object_pairs_num, FLAGS.g_logits))

        print("discriminator g summation...")
        g_sum = tf.reduce_sum(g_out, axis=1)

        print("discriminator f func in progress...")
        f_layer1 = tf.layers.dense(
            inputs=g_sum,
            units=FLAGS.f_hidden1,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
            bias_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
            activity_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_scale),
            trainable=True,
            reuse=None,
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
            reuse=None,
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
            reuse=None,
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
            reuse=None,
            name="supervised_layer"
            )

        print("discriminator f func in progress... supervised logits complete")

        return logits, supervised_logits


    def _gan_loss(self, logits_real, logits_fake, supervised_logits, label, use_features=False):
        supervised_loss = tf.losses.softmax_cross_entropy(label, supervised_logits)
        discriminator_loss = tf.reduce_mean(logits_real - logits_fake) + supervised_loss
        gen_loss = tf.reduce_mean(logits_fake)
        tf.summary.scalar('discriminator_loss', discriminator_loss)
        tf.summary.scalar('gen_loss', gen_loss)
        tf.summary.scalar('supervised_loss', supervised_loss)
        return discriminator_loss, gen_loss


    def _create_network(self, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
        print("Setting up model...")
        self.gen_train_op = []
        self.disc_train_op = []
        for gpu in ["/gpu:0","/gpu:1","/gpu:2","/gpu:3"]:
            self.train_batch = tf.placeholder(
                dtype=tf.float32, 
                shape=[FLAGS.batch_size, FLAGS.max_document_length, FLAGS.embed_dim])
            
            self.label_indices = tf.placeholder(
                dtype=tf.int32, 
                shape=[FLAGS.batch_size,])

            print(gpu + "  real data pairing..")
            real_pairs = self.preproc.pairing(self.train_batch)

            print(gpu + "   create generator...")
            with tf.variable_scope("generator") as g_scope:
                if gpu != "/gpu:0": 
                    print("reuse generator")
                    g_scope.reuse_variables()
                self._create_generator(FLAGS.batch_size)
                
            
            print(gpu + "   fake data pairing..")
            fake_pairs = self.preproc.pairing(self.gen_data)

            print(gpu + "   building discriminator for real data...")
            with tf.variable_scope("discriminator") as d_scope:
                if gpu != "/gpu:0":
                    print("reuse discriminator") 
                    d_scope.reuse_variables()
                logits_real, logits_supervised = self._discriminator(real_pairs)
                d_scope.reuse_variables()

                print(gpu + "   building discriminator for fake data...")
                logits_fake, _ = self._discriminator(fake_pairs)

            print("/cpu:0   building gan loss graph...")
            labels = tf.one_hot(indices=self.label_indices, depth=3, on_value=1.0, off_value=0.0)
            self.disc_loss, self.gen_loss = self._gan_loss(logits_real, logits_fake, logits_supervised, labels)


            self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
            self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]

            print(list(map(lambda x: x.op.name, self.generator_variables)))
            print(list(map(lambda x: x.op.name, self.discriminator_variables)))

            optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

            print(gpu + "   building train op")
            # self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
            # self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)
            self.gen_grads = optim.compute_gradients(self.gen_loss, self.generator_variables)
            self.disc_grads = optim.compute_gradients(self.disc_loss, self.discriminator_variables)

            self.gen_train_op.append(optim.apply_gradients(self.gen_grads))
            self.disc_train_op.append(optim.apply_gradients(self.disc_grads))

            tf.get_variable_scope().reuse_variables()



        # self.gen_average_grads = self.average_grads(self.gen_grads)
        # self.disc_average_grads = self.average_grads(self.disc_grads)

        # self.gen_train_op = optim.apply_gradients(self.gen_average_grads)
        # self.disc_train_op = optim.apply_gradients(self.disc_average_grads)
            
        self.saver = tf.train.Saver(self.generator_variables)


    def average_grads(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)

        return average_grads


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
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        for itr in range(1, max_iterations):
            train_data, indices = self.preproc.get_batch(self.data, itr-1)
            feed_dict = {self.train_batch: train_data, self.label_indices: indices}

            if itr < 25 or itr % 500 == 0:
                critic_itrs = 25
            else:
                critic_itrs = self.critic_iterations

            for critic_itr in range(critic_itrs):
                # print("discriminator critic: ", critic_itr)
                self.sess.run(self.disc_train_op, feed_dict)
                self.sess.run(clip_discriminator_var_op, feed_dict)
            
            # print("generator update")
            for i in range(4):
                summary, _ = self.sess.run([merged, self.gen_train_op[i], feed_dict)

            if itr % 100 == 0:
                g_loss_val, d_loss_val = self.sess.run(
                    [self.gen_loss, self.disc_loss], feed_dict)
                self.saver.save(self.sess, "gs://wgan/logs/wgan")
                summary_writer.add_summary(summary, itr)
                print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))


    def open_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        print("train ready")


def main(argv=None):
    gan = WassersteinGAN(critic_iterations=5)
    gan._create_network()                
    gan.train_model(1000)
    gan.sess.close()

if __name__ == "__main__":
    tf.app.run()
