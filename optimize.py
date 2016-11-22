import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from network import Network
import pdb

data = np.load('data/textVectors.npz')['X']
nSamples, nWords, nDim = data.shape

def getBatch(data, batchSize):
    batch = np.zeros((batchSize, data.shape[1], data.shape[2]), dtype=np.float32)
    for i in range(batchSize):
        idx = np.random.randint(data.shape[0])
        batch[i,:,:] = data[idx,:,:]
    return batch, 1

bindingData = np.load('data/textBindings.npz')['X']
nMemories = nBindings = bindingData.shape[0]
binding_length = bindingData.shape[1]

# Parameters
learningRate = 1e-2
trainingIters = 1000
batchSize = 4

# Network Parameters
nInput = nDim
nSteps = nWords
nHidden = 2 ** 7
displayStep = nSteps

# Network
bindings = tf.constant(bindingData.astype(np.float32))
memories = tf.Variable(tf.zeros([batchSize, nMemories, nDim]))
network  = Network(nInput, nHidden, nBindings, nMemories, bindings, binding_length, batchSize)
hstate   = network.get_new_state(batchSize)

# tf Graph input
x = tf.placeholder("float", [batchSize, nInput])
#y = tf.placeholder("float", [None, n_classes])

# Step through
for i in range(nSteps):
    hstate, memories = network.step(hstate, memories, x)


# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batchSize < trainingIters:
        batchX, batchY = getBatch(data, batchSize)

        for i in range(nSteps):
            batchXi = batchX[:,i,:]
            inputX = tf.constant(batchXi)
            hstate, memories = network.step(hstate, memories, inputX, batchSize)

        memory = tf.slice(memories, [0, 0, 0], [-1, 1, -1]) # extract the first memory
        # 4x1x1024
        memory = tf.reshape(memory, [batchSize, binding_length])
        cost = tf.constant(0)
        for test in range(2):
            break #FIXME
            unbound = tf.mul(test.inverse, hdv)
            result = tf.einsum("kj,kj->k", unbound, test.identity)
            cost += tf.constant(binding_length) - tf.reduce_mean(result)

        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
        sess.run(optimizer)

        # Run optimization op (backprop)
        #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if False and step % displayStep == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
