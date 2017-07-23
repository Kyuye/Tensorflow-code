
import tensorflow as tf
import pandas
import numpy as np
import re 
from pandas import Series, DataFrame 
import string 

def dense_layer(
    inputs, 
    units, 
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.9),
    bias_regularizer=tf.contrib.layers.l2_regularizer(0.9),
    activity_regularizer=tf.contrib.layers.l2_regularizer(0.9),
    trainable=True,
    reuse=False, 
    name=None):
    return tf.layers.dense(
            inputs=inputs,
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            name=name,
            reuse=reuse)


def rand(shape):
    random_uniform = tf.random_uniform(shape=shape, minval=-1, maxval=1, dtype=tf.float32)
    return tf.unstack(random_uniform, axis=1)

def LSTM_Wo(shape, reuse):
    var = tf.get_variable(
        name="Wo",
        shape=shape,
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer())
    return tf.unstack(var)


def LSTM_bo(shape, reuse):
    var = tf.get_variable(
        name="bo",
        shape=shape,
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer())
    return tf.unstack(var)

def one_hot(indices):
    return tf.one_hot(indices=indices, depth=3, on_value=1.0, off_value=0.0)

def filter_word(target,sent):
    filtered = sent
    for remove in target:
         filtered = list(map(lambda s: filter(lambda w: w.find(remove), filter(None, s)), filtered))
    return filtered

def filter_printable(sent):
    printable_set = set(string.printable)
    return list(map(lambda s: filter(lambda w: w[0] in printable_set, s), sent))

def filter_none(sent):
    return filter(None, sent)

