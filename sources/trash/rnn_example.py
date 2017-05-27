"""
* this is rnn practice code
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
    def init(self):
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

        self.ts_batch_y = tf.reshape(ts_y[:-1], sz_batch)
        self.ts_batch_y_ = tf.reshape(ts_y[1:], sz_batch)


    def build_batch(self):
        batch_set = [self.ts_batch_y, self.ts_batch_y_]
        self.b_train, self.b_label = tf.train.batch(batch_set, CONST.batch_size, enqueue_many=True)

    def set_variables(self):
        sz_weight = (CONST.recurrent, CONST.state_size, CONST.input_vector_size)
        sz_bias = (CONST.recurrent, 1, CONST.input_vector_size)
        linear_w = tf.Variable(tf.truncated_normal(sz_weight))
        linear_b = tf.Variable(tf.zeros(sz_bias))

        self.linear_w = tf.unstack(linear_w)
        self.linear_b = tf.unstack(linear_b)

    def build_model(self):
        rnn_cell = tf.nn.rnn_cell.RNNCell(CONST.state_size)
        self.input_set = tf.unstack(self.b_train, axis=1)
        self.label_set = tf.unstack(self.b_label, axis=1)
        self.output, _ = tf.nn.rnn(rnn_cell, self.input_set, dtype=tf.float32)
        self.pred = tf.matmul(self.output, self.linear_w) + self.linear_b

    def build_train(self):
        self.loss = 0
        for i in range(CONST.recurrent):
            self.loss += self._mean_square_error(self.pred[i], self.label_set[i])

        self.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(self.loss)

    def mean_square_error(self, batch, label):
        return tf.reduce_mean(tf.pow(batch - label, 2))


def main(_):
    rnn = RNN()
    if CONST.is_training_mode is True:
        rnn.training()
    else:
        rnn.prediction()

if __name__ == "__main__":
    tf.app.run()
    
