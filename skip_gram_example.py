"""
* this is the skip gram practice code
"""

import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("dict_size", 50000, "")
CONSTANT.DEFINE_integer("embed_size", 200, "")
CONSTANT.DEFINE_integer("num_sampled", 64, "")
CONSTANT.DEFINE_integer("learning_rate", 1e-4, "")
CONSTANT.DEFINE_integer("epoch", 100, "")
CONST = CONSTANT.FLAGS


class SkipGram(object):
    """
     * skip gram class
    """
    def init(self):
        self._build_skipgram()
        self._initialization()

    def training(self):
        """
         * run the train
        """

        for step in xrange(CONST.epoch):
            loss = self._run_train()
            if step % 10 == 0:
                print loss

        self._close_session()

    def prediction(self):
        """
         * run the prediction
        """
        embeddings = self._run_prediction()
        print embeddings[0]
        self._close_session()


    def run_train(self):
        _, loss = self.sess.run([self.optimizer, self.loss])
        return loss

    def run_prediction(self):
        embeddings = self._get_embeddings()
        return self.sess.run(embeddings)

  
    def gen_onehots(self):
        indices = tf.cast(tf.transpose([range(CONST.dict_size), range(CONST.dict_size)]), tf.int64)
        values = tf.ones((CONST.dict_size), dtype=tf.int64)
        shape = (CONST.dict_size, CONST.dict_size)
        st_onehots = tf.SparseTensor(indices=indices, values=values, shape=shape)
        self.ts_onehots = tf.sparse_tensor_to_dense(st_onehots)

        
    def set_variables(self):
        self.embed_dict = tf.Variable(tf.random_uniform((CONST.dict_size, CONST.embed_size)))

    def build_skipgram(self):
        sz_weights = (CONST.dict_size, CONST.embed_size)

        init_embeddings = tf.random_uniform(sz_weights, -1, 1)
        self.embeddings = tf.Variable(init_embeddings)

        stddev = 1.0/tf.sqrt(tf.cast(CONST.embed_size, tf.float32))
        init_nce_weight = tf.truncated_normal(sz_weights, stddev=stddev)
        w_nce = tf.Variable(init_nce_weight)

        init_nce_bias = tf.zeros((CONST.dict_size))
        nce_bias = tf.Variable(init_nce_bias)

        x_word = tf.constant([1, 2, 3, 4, 5])
        y_word = tf.constant([[4], [5], [6], [7], [8]])
        embeds = tf.nn.embedding_lookup(self.embeddings, x_word)

        self.loss = tf.nn.nce_loss(
            weights=w_nce,
            biases=nce_bias,
            inputs=embeds,
            labels=y_word,
            num_sampled=CONST.num_sampled,
            num_classes=CONST.dict_size
        )

        self.optimizer = tf.train.GradientDescentOptimizer(CONST.learning_rate).minimize(self.loss)

   
    def get_embeddings(self):
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        return self.embeddings/norm

    
    def initialization(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    
    def close_session(self):
        self.sess.close()

def main(_):
    """
    * this is main function of this project
    """
    skip_gram = SkipGram()
    #skip_gram.training()
    skip_gram.prediction()
    print "process done"

if __name__ == "__main__":
    tf.app.run()
