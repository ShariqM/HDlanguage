import tensorflow as tf
import numpy as np
import layers
import pdb
from helper import complexMultiply

class Network(object):
    def __init__(self, batchSize, n_input, n_HD_input, n_hidden, n_bindings, n_memories, bindings):
        self.rnn_layer   = layers.RNN_Layer(n_input, n_hidden)
        self.read_layer  = layers.Linear_Layer(n_hidden, n_bindings)
        self.write_layer = layers.Linear_Layer(n_hidden, n_memories)

        self.n_input = n_input
        self.n_HD_input = n_HD_input
        self.batchSize = batchSize
        self.bindingsReal = bindings[0]
        self.bindingsImag = bindings[1]
        self.n_memories = n_memories

        self.writing_softmax = []


    def step(self, hstate, memories, x_input, hdInput):
        hstate = self.rnn_layer.step(hstate, x_input)

        scaleParam = tf.constant(1.0)
        reading_raw = self.read_layer.output(hstate)
        reading_raw = tf.scalar_mul(scaleParam, reading_raw)
        reading_softmax = tf.nn.softmax(reading_raw)
        self.reading_softmax = reading_softmax

        writing_raw = self.write_layer.output(hstate)
        writing_raw = tf.scalar_mul(scaleParam, writing_raw)
        writing_softmax = tf.nn.softmax(writing_raw)
        self.writing_softmax.append(writing_softmax)

        writing_softmax_exp = tf.reshape(writing_softmax, [self.batchSize, self.n_memories, 1])
        writing_softmax_exp = tf.tile(writing_softmax_exp, [1, 1, self.n_HD_input])

        bindingReal = tf.matmul(reading_softmax, self.bindingsReal) # ki,ij
        bindingImag = tf.matmul(reading_softmax, self.bindingsImag) # ki,ij

        hdInputReal, hdInputImag = hdInput
        boundedXReal, boundedXImag = complexMultiply(bindingReal, bindingImag,
                                                     hdInputReal, hdInputImag)

        boundedXRealExp = tf.reshape(boundedXReal, [self.batchSize, 1, self.n_HD_input])
        boundedXReals = tf.tile(boundedXRealExp, [1, self.n_memories, 1])

        boundedXImagExp = tf.reshape(boundedXImag, [self.batchSize, 1, self.n_HD_input])
        boundedXImags = tf.tile(boundedXImagExp, [1, self.n_memories, 1])

        memoriesReal, memoriesImag = memories
        memoriesReal = memoriesReal + tf.mul(boundedXReals, writing_softmax_exp)
        memoriesImag = memoriesImag + tf.mul(boundedXImags, writing_softmax_exp)
        return hstate, (memoriesReal, memoriesImag)

    def get_reading_softmax(self):
        return self.reading_softmax

    def get_writing_softmax(self):
        return self.writing_softmax

    def get_new_state(self, n_batch):
        return self.rnn_layer.get_new_state(n_batch)
