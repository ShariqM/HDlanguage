import tensorflow as tf
import numpy as np
import layers
import pdb

class Network(object):
    def __init__(self, n_input, n_hidden, n_bindings, n_memories, bindings, batchSize, binding_length):
        self.rnn_layer   = layers.RNN_Layer(n_input, n_hidden)
        self.read_layer  = layers.Linear_Layer(n_hidden, n_bindings)
        self.write_layer = layers.Linear_Layer(n_hidden, n_memories)

        self.bindings = bindings
        self.binding_length = binding_length
        self.n_memories = n_memories

    def step(self, hstate, memories, x_input):
        hstate = self.rnn_layer.step(hstate, x_input)

        reading_raw = self.read_layer.output(hstate)
        reading_softmax = tf.nn.softmax(reading_raw)

        writing_raw = self.write_layer.output(hstate)
        writing_softmax = tf.nn.softmax(writing_raw)
        writing_softmax_exp = tf.reshape(writing_softmax, [self.batchSize, self.n_memories, 1])
        writing_softmax_exp = tf.tile(writing_softmax_exp, [1, 1, self.binding_length])

        binding = tf.matmul(reading_softmax, self.bindings) # ki,ij
        boundedX = tf.mul(binding, x_input)
        boundedXexp = tf.reshape(boundedX, [self.batchSize, 1, self.binding_length])
        boundedXs = tf.tile(boundedXexp, [1, 10, 1])

        memories = memories + tf.mul(boundedXs, writing_softmax_exp)
        return hstate, memories

    def get_new_state(self, n_batch):
        return self.rnn_layer.get_new_state(n_batch)
