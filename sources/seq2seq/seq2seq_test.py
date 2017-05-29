
import tensorflow as tf


PAD = 0
EOS = 1
tf.reset_default_graph()
sess = tf.InteractiveSession() 

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2

encoder_inputs = tf.placeholder(tf.int32, (None, None))
encoder_inputs_length = tf.placeholder(tf.int32)
decoder_targets = tf.placeholder(tf.int32)

embeddings = tf.Variable(tf.random_uniform(shape=(vocab_size, input_embedding_size), minval=-1.0, maxval=1.0), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

encoder_cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
encoder_bw_outputs),
(encoder_fw_final_state,
encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(
    cell_fw=encoder_cell,
    cell_bw=encoder_cell,
    inputs=encoder_inputs_embedded,
    sequence_length=encoder_inputs_length,
    dtype=tf.float32, time_major=True))

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)


decoder_cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3

W = tf.Variable(tf.random_uniform((decoder_hidden_units, vocab_size), -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros((vocab_size)), dtype=tf.float32)

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones((batch_size), dtype=tf.int32)
pad_time_slice = tf.zeros((batch_size), dtype=tf.int32)

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None

    return (initial_elements_finished,
    initial_input,
    initial_cell_state,
    initial_cell_output,
    initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths)

    finished = tf.reduce_all(elements_finished)
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)

    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
    input,
    state,
    output,
    loop_state)