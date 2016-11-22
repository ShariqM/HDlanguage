# Define weights
weights = {
        'HB': tf.Variable(tf.random_normal([nHidden, nBindings]))
        'HM': tf.Variable(tf.random_normal([nHidden, nMemories]))
}

biases = {
        'HR': tf.Variable(tf.random_normal([nBindings]))
        'HW': tf.Variable(tf.random_normal([nMemories]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, nInput])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, nSteps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    readingRawVector = tf.matmul(outputs[-1], weights['HR']) + biases['HR']
    writeRawVector = tf.matmul(outputs[-1], weights['HW']) + biases['HW']

    return readingRawVector, writeRawVector

readingRawVector, writeRawVector = RNN(x, weights, biases)
readingSoftMaxVector = tf.nn.softmax(readingRawVector)
bindingVector = tf.einsum("i,ij", readingSoftMaxVector, bindings)

20k


