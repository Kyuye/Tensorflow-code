
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import EmbeddingWrapper
from tensorflow.contrib.rnn import static_rnn
import tensorflow.contrib.seq2seq as seq2seq
# import attention_wrapper as wrapper


tf.reset_default_graph()

sess = tf.InteractiveSession()

embedding_dim = 20
input_seq_length = 30
output_seq_length = 30
input_vocab_size = 100000

## Place holders

encode_input = [tf.placeholder(tf.int32, shape=(None,), name = "ei_%i" %i) for i in range(input_seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,), name = "l_%i" %i) for i in range(output_seq_length)]

decode_input = [tf.zeros_like(encode_input[0], dtype=np.int32, name="GO")] + labels[:-1]



############ Encoder
lstm_cell = BasicLSTMCell(embedding_dim)
encoder_cell = EmbeddingWrapper(lstm_cell, embedding_classes=input_vocab_size, embedding_size=embedding_dim)
encoder_outputs, encoder_state = static_rnn(encoder_cell, encode_input, dtype=tf.float32) 

############ Decoder
# Attention Mechanisms. Bahdanau is additive style attention
attn_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=input_seq_length, memory=encoder_outputs, normalize=False, name='BahdanauAttention')
lstm_cell_decoder = BasicLSTMCell(embedding_dim)

# Attention Wrapper: adds the attention mechanism to the cell
attn_cell = seq2seq.DynamicAttentionWrapper(
    cell = lstm_cell_decoder, 
    attention_mechanism = attn_mech, # Instance of AttentionMechanism
    attention_size = embedding_dim, # Int, depth of attention (output) tensor
    name="attention_wrapper")


# Decoder setup
decoder = seq2seq.BasicDecoder(
          cell = lstm_cell_decoder,
          helper = helper, # A Helper instance
          initial_state = encoder_state, # initial state of decoder
          output_layer = None) # instance of tf.layers.Layer, like Dense

# Perform dynamic decoding with decoder object
outputs, final_state = seq2seq.dynamic_decode(decoder)