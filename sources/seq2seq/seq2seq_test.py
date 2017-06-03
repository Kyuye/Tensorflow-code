
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

tf.reset_default_graph()

sess = tf.Session()

# encoder == RNN(EmbeddingWrapper(cell))
lstm_cell = BasicLSTMCell(
    num_units=embedding_dim)

encoder_cell = EmbeddingWrapper(
    cell=lstm_cell,
    embedding_classes=input_vocab_size,
    embedding_size=embedding_dim)

encoder_outputs, encoder_state = static_rnn(
    cell=encoder_cell,
    inputs=encode_input,
    dtype=tf.float32)

# Attention == 
attn_mech = seq2seq.BahdanauAttention(
    num_units=input_seq_length,
    memory=encoder_outputs,
    normalize=False,
    name='BahdanauAttention')

lstm_cell_decoder = BasicLSTMCell(embedding_dim)

attn_cell = seq2seq.DynamicAttentionWrapper(
    cell=lstm_cell_decoder,
    attention_mechanism=attn_mech,
    attention_size=embedding_dim,
    name="attention_wrapper")

decoder = seq2seq.BasicDecoder(
    cell=lstm_cell_decoder,
    helper=helper,
    initial_state=encoder_state,
    output_layer=None)

outputs, final_state = seq2seq.dynamic_decode(decoder)

sess.close()
