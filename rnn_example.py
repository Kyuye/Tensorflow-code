"""
* this is rnn example
"""

import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("epoch", 100, "")
CONSTANT.DEFINE_integer("samp", 3000000, "")
CONSTANT.DEFINE_integer("state_size", 10, "")
CONSTANT.DEFINE_integer("steps", 10, "")
CONSTANT.DEFINE_integer("input_size", 300, "")
CONSTANT.DEFINE_float("learning_rate", 0.001, "")
CONSTANT.DEFINE_string("ckpt_dir", "./checkpoint/rnn.ckpt", "")
CONSTANT.DEFINE_string("tensorboard_dir", "./tensorboard", "")
CONSTANT.DEFINE_integer("batch", 100, "")
CONST = CONSTANT.FLAGS

class RNN(object):
    """
     * RNN model
    """
    def __init__(self):
        self._gen_sim_data()
        self._build_batch()
        self._set_variables()
        self._build_model()
        self._save_model()
        self._build_train()
        self._initialize()

    def training(self):
        for step in range(CONST.epoch):
            loss = self._run_train()
            if step % 10 == 0:
                print("step: ", step)
                print("loss: ", loss)

        self._write_checkpoint(CONST.ckpt_dir)

    def prediction(self):
        self._run_pred()
        self._close_session()

    def run_train(self):
        self.sess.run(tf.global_variables_initializer())
        _, loss = self.sess.run([self.train, self.loss])
        return loss

    def run_pred(self):
        self._restore_checkpoint(CONST.ckpt_dir)
        return self.sess.run(self.pred)

    def save_model(self):
        self.saver = tf.train.Saver()

    def write_checkpoint(self, directory):
        self.saver.save(self.sess, directory)

    def restore_checkpoint(self, directory):
        self.saver.restore(self.sess, directory)

    def initialize(self):
        self.sess = tf.Session()
        self.coord = tf.train.Coordinator()
        self.thread = tf.train.start_queue_runners(self.sess, self.coord)

    def close_session(self):
        self.coord.request_stop()
        self.coord.join(self.thread)
        self.sess.close()

    def gen_sim_data(self):
        ts_x = tf.constant([i for i in range(CONST.samples+1)], dtype=tf.float32)
        ts_y = tf.sin(ts_x*0.01)

        sz_batch = (
            int(CONST.samples/(CONST.recurrent*CONST.input_vector_size)),
            CONST.steps,
            CONST.input_size)

        cls.ts_batch_y = tf.reshape(ts_y[:-1], sz_batch)
        cls.ts_batch_y_ = tf.reshape(ts_y[1:], sz_batch)


    def build_batch(cls):
        batch_set = [cls.ts_batch_y, cls.ts_batch_y_]
        cls.b_train, cls.b_label = tf.train.batch(batch_set, CONST.batch_size, enqueue_many=True)

    def set_variables(cls):
        sz_weight = (CONST.recurrent, CONST.state_size, CONST.input_vector_size)
        sz_bias = (CONST.recurrent, 1, CONST.input_vector_size)
        linear_w = tf.Variable(tf.truncated_normal(sz_weight))
        linear_b = tf.Variable(tf.zeros(sz_bias))

        cls.linear_w = tf.unstack(linear_w)
        cls.linear_b = tf.unstack(linear_b)

    def build_model(self):
        rnn_cell = tf.nn.rnn_cell.RNNCell(CONST.state_size)
        cls.input_set = tf.unstack(cls.b_train, axis=1)
        cls.label_set = tf.unstack(cls.b_label, axis=1)
        cls.output, _ = tf.nn.rnn(rnn_cell, cls.input_set, dtype=tf.float32)
        cls.pred = tf.matmul(cls.output, cls.linear_w) + cls.linear_b

    def build_train(self):
        self.loss = 0
        for i in range(CONST.recurrent):
            self.loss += self._mean_square_error(self.pred[i], self.label_set[i])

        cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)

    def mean_square_error(cls, batch, label):
        return tf.reduce_mean(tf.pow(batch - label, 2))


def main(_):
    rnn = RNN()
    if CONST.is_training_mode is True:
        rnn.training()
    else:
        rnn.prediction()

if __name__ == "__main__":
    tf.app.run()
    
