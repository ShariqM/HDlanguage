import tensorflow as tf

def init_weights(n_input, n_unit):
    return tf.truncated_normal([n_input, n_unit], 0.0,
        tf.sqrt(2.0)/tf.sqrt(tf.cast(n_input + n_unit, tf.float32)))
    #return tf.truncated_normal([n_input, n_unit], 0.0,
        #tf.sqrt(2.0)/tf.sqrt(tf.cast(n_input + n_unit, tf.float32)))

class Linear_Layer(object):
    def __init__(self, n_input, n_output):
        self.W = tf.Variable(init_weights(n_input, n_output), name='W')
        self.b = tf.Variable(tf.zeros([n_output]), name='b')

    def output(self, x):
        return tf.nn.xw_plus_b(x, self.W, self.b)

class RNN_Layer(object):
    def __init__(self, n_input, n_unit):
        self.n_unit = n_unit
        self.IH = tf.Variable(init_weights(n_input, n_unit), name='IH')
        self.HH = tf.Variable(init_weights(n_unit, n_unit), name='HH')
        self.b = tf.Variable(tf.zeros([1, n_unit]), name='b')

    def get_new_state(self, n_batch):
        #return tf.Variable(tf.zeros(self.n_unit), trainable=False, name='h' )
        return tf.Variable(tf.random_normal([n_batch, self.n_unit]),
                           trainable=False, name='h' )
    def step(self, h, x):
        """Updates returns the state updated by input x"""
        g = tf.matmul(x, self.IH)
        h = tf.tanh(tf.matmul(x, self.IH) + tf.matmul(h, self.HH) + self.b)
        return h
