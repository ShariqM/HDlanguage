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
batchSize = 2

# Network Parameters
nInput = nDim
nSteps = nWords
nHidden = 2 ** 7
displayStep = nSteps

# Network
bindings = tf.constant(bindingData.astype(np.float32))
memories = tf.Variable(tf.zeros([batchSize, nMemories, nDim]))
network  = Network(batchSize, nInput, nHidden, nBindings, nMemories, bindings)
hstate   = network.get_new_state(batchSize)

# tf Graph input
x = tf.placeholder("float", [batchSize, nSteps, nInput])
invBinding = tf.placeholder("float", [batchSize, nInput])
target = tf.placeholder("float", [batchSize, nInput])

# Permuting batch_size and n_steps
x2 = tf.transpose(x, [1, 0, 2])
# Reshaping to (n_steps*batch_size, n_input)
x2 = tf.reshape(x2, [-1, nInput])
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x2 = tf.split(0, nSteps, x2)

# Step through
for i in range(nSteps):
    hstate, memories = network.step(hstate, memories, x2[i])
memoryExp = tf.slice(memories, [0, 0, 0], [-1, 1, -1]) # extract the first memory
memory = tf.reshape(memoryExp, [batchSize, nInput])

cost = tf.constant(0.0)
for i in range(2):
    unbound = tf.mul(invBinding, memory)
    result = tf.einsum("kj,kj->k", unbound, target)
    cost += tf.constant(float(nInput)) - tf.reduce_mean(result)

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batchSize < trainingIters:
        batchX, batchY = getBatch(data, batchSize)

        binvBinding = np.random.randn(batchSize, nInput)
        btarget = np.random.randn(batchSize, nInput)

        sess.run(optimizer, feed_dict={x: batchX, invBinding: binvBinding, target: btarget})

        #a, b, c = tf.constant(batchX), tf.constant(invBinding), tf.constant(target)
        #cost = sess.run(cost, feed_dict={x: a, invBinding: b, target: c})
        #cost = sess.run(cost)
        costVal = sess.run(cost, feed_dict={x: batchX, invBinding: binvBinding, target: btarget})
        print ('cost', costVal)
        #memVal = sess.run(hstate, feed_dict={x: batchX})
        #print ('mem', memVal)


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
    #print ("Testing Accuracy:", \
        #sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
