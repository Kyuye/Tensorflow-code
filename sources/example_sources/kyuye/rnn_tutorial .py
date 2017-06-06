"""
 * rnn turtorial 
"""


import tensorflow as tf  
import time

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("samples", 1000,"simulation data samples")
CONSTANT.DEFINE_integer("time_step ", 5, "time step in rnn")
CONSTANT.DEFINE_integer("vec_size", 1, "input vector size into rnn")
CONSTANT.DEFINE_integer("batch_size",  10, "minibatch size for training")
CONSTANT.DEFINE_integer("state_size", 15, "state size in rnn")
CONSTANT.DEFINE_integer("learning_rate", 0.01, "learning rate for opimizer"
CONST = CONSTANT.FLAGS 

class rnn_example(object):
    """
     * rnn example module
    """
# constructor
    def __init__(self):
        self._gen_sim_data()
        self._build_batch()
        self._build_model()
        self._build_train()
        self._initialize()
        
        #self._pack_test()
        
    def run(self):
    
    """
     * run the rnn model
    """

    # initializing the variables
    self.sess.run(tf.global_variables_initializer())

    # printing train, loss in 1000 times iteration 
    for i in range(1000):
        _, loss = self.sess.run([self.train, self.loss])
        
        # printing loss at every 20 steps in 1000 steps
        if i % 20 == 0 :
            print("loss:", loss)
    
    self._close_session()
     
        
@classmethod # running session 
def _run_session(cls, run_graph)
    output = cls.sess.run(run_graph)
    return output 

@classmethod # initializing : opening tf session & start thread
def _initialize(cls):
    cls.sess = tf.Session()
    cls.coord = tf.train.Coordinator() # class coorniator()
    cls.thread = tf.train.start_queue_runner(cls.sess, cls,coord)

@classmethod # closing session 
def _close_session(cls):
    cls.coord.request_stop()
    cls.coord.join(cls.thread) # Wait for all the threads to terminate.
    cls.sess.close()

@classmethod # generarting data
def _gen_sim_data(cls):
    cls.ts_x = tf.constant([i for i in range(CONST.samples+1)],dtype=tf.float32)
    #?
    cls.ts_y = tf.sin(cls.ts_x *0.1)

    # Making batch shape
    sp_batch = (int(CONST.samples/CONST.time_step), CONST.time_step,  CONST.vec_size)
    
    # the shape of gerated data ts_x,ts_y are changed by batch shape 
    # tf.reshape (current shape, expected shape)
    cls.fbatch_input = tf.reshape(ts_y[:-1], sp_batch)
    cls.fbatch_label = tf.reshape(ts_y[1:],sp_batch)

@classmethod # building rnn batch
def _build_batch(cls):
    # Making full batch set
    cls.batch_set = [cls.fbatch_input, cls.fbatch_label]
    # Creates mini batch set of tensors to make package  tf.train.batch(dictionary of tensors,batch_size)
    # batch size = mini batch size
    cls.mb_train, cls.mb_label = tf.train.batch(batch_set, CONST.batch_size, enqueue_many=True)

@classmethod # buidling rnn model
def _build model(cls):
    rnn_cell = tf.contrib.rnn.BasicRNNCell(CONST.state_size)
    # MAKING RNN tf.contrib.rnn.static_rnn(cell, inputs), return : output, state(= _))
    cls.output, cls.state = tf.contrib.rnn.static_rnn(rnn_cell, tf.unstack(cls.mb_train, axis=1), dype=tf.float32)
    # Making weight = tf.Variable(tf.truncated_normal([size]))
    cls.output_w = tf.Variale(tf.truncated_normal([CONST.time_step, CONST.state_size, CONST.vec_size]))
    cls.output_b = tf.Variable(tf.zeros([CONST.vec_sisze]))

    cls.pred = tf.matmul(cls.output, cls.output_w)+ output_b

    print("the shpae of output_w:", cls.sess.run(tf.shape(cls.output_w)))
    print("the shape of output_b:", cls.sess.run(tf.shape(cls.outputb)))
    print("the shape of output:", cls.sess.run(tf.shape(cls.output)))

@classmethod #building training model
def _build_train(cls):
     cls.loss = 0  # initialize to 0 
     for i in ranage(CONST.time_step)
        cls.loss += tf.losses.mean_squared_error(tf.unstack(cls.mb_label,axis=1)[i],cls.pred[i])
    # Training to minimize the loss using AdamOptimizer 
    cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)



# tensorflow main function 
def main(_):
    """
     * code begins here
    """
    # return the current processor time as a floating point number expressed in seconds.
    tic = time.clock()  
    # 'rnn' is 'class rnn_example ' (생성자, constructor)
    rnn = rnn_example() 
    # running the class rnn_example
    rnn.run()
    toc = time.clock()
    print("total process time:", toc-tic)

# python main function
if __name__=="__main__"
    tf.app.run()

