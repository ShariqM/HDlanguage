import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from network import Network
import pdb
from helper import complexMultiply, getBatch

allData = np.load('data/textVectors.npz')
inputData = allData['V']
bindingsReal, bindingsImag = allData['BR'], allData['BI']
hdInput = allData['HDVR'], allData['HDVI']
invBindings = allData['iBR'], allData['iBI']
allWordsReal, allWordsImag = allData['AWR'], allData['AWI']
targets = allData['T']

nSamples, nWords, nDim = inputData.shape
nHDDim = hdInput[0].shape[2]
nQuestions = invBindings[0].shape[1]
nMemories = nBindings = bindingsReal.shape[0]
totWords = targets.shape[2]

# Parameters
learningRate = 1e-1
trainingIters = 10000
batchSize = 8

# Network Parameters
nInput = nDim
nHDInput = nHDDim
nSteps = nWords
nHidden = 2 ** 7
displayStep = nSteps

# Network
bindingsRealTF = tf.constant(bindingsReal.astype(np.float32))
bindingsImagTF = tf.constant(bindingsImag.astype(np.float32))
bindingsTF = (bindingsRealTF, bindingsImagTF)

allWordsRealExp = np.reshape(allWordsReal, [1, totWords, nInput])
allWordsReals   = np.tile(allWordsRealExp, [batchSize, 1, 1])
allWordsImagExp = np.reshape(allWordsImag, [1, totWords, nInput])
allWordsImags   = np.tile(allWordsImagExp, [batchSize, 1, 1])

allWordsRealTF = tf.constant(allWordsReals.astype(np.float32))
allWordsImagTF = tf.constant(allWordsImag.astype(np.float32))
allWordsTF = (allWordsRealTF, allWordsImagTF)

memoriesRealTF = tf.Variable(tf.zeros([batchSize, nMemories, nDim]))
memoriesImagTF = tf.Variable(tf.zeros([batchSize, nMemories, nDim]))
memoriesTF = (memoriesRealTF, memoriesImagTF)

network  = Network(batchSize, nInput, nHidden, nBindings, nMemories, bindingsTF)
hstateTF   = network.get_new_state(batchSize)

# tf Graph input
xTF = tf.placeholder("float", [batchSize, nSteps, nInput])

xHDRealTF = tf.placeholder("float", [batchSize, nSteps, nHDInput])
xHDImagTF = tf.placeholder("float", [batchSize, nSteps, nHDInput])
xHDTF = (xHDRealTF, xHDImagTF)

invBindingRealTF = tf.placeholder("float", [batchSize, nQuestions, nHDInput])
invBindingImagTF = tf.placeholder("float", [batchSize, nQuestions, nHDInput])
invBindingTF = (invBindingRealTF, invBindingRealTF)

targetsTF = tf.placeholder("float", [batchSize, nQuestions, totWords])

def stepify(x, nInput, nSections):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    return tf.split(0, nSections, x)

def unroll(x, xHD, memories, hstate):
    x   = stepify(x, nInput, nSteps)
    xHD = (stepify(xHD[0], nHDInput, nSteps), stepify(xHD[1], nHDInput, nSteps))

    # Step through
    for i in range(nSteps):
        xHDi = (xHD[0][i], xHD[1][i])
        hstate, memories = network.step(hstate, memories, x[i], xHDi)

    memoriesReal, memoriesImag = memories
    memoryRealExp = tf.slice(memoriesReal, [0, 0, 0], [-1, 1, -1])
    memoryReal = tf.reshape(memoryRealExp, [batchSize, nHDInput])
    memoryImagExp = tf.slice(memoriesReal, [0, 0, 0], [-1, 1, -1])
    memoryImag = tf.reshape(memoryImagExp, [batchSize, nHDInput])

    return (memoryReal, memoryImag)

memory = unroll(xTF, xHDTF, memoriesTF, hstateTF)

def computeCost(invBinding, allWords, targets, memory):
    invBinding = (stepify(invBinding[0], nHDInput, nQuestions),
                  stepify(invBinding[1], nHDInput, nQuestions))
    targets    = stepify(targets, totWords, nQuestions)

    invBindingReal, invBindingImag = invBinding
    memoryReal, memoryImag = memory
    cost = tf.constant(0.0)
    for i in range(nQuestions):
        unboundReal, unboundImag = complexMultiply(invBindingReal[i],
                                                   invBindingImag[i],
                                                   memoryReal, memoryImag)
        # unboundReal - batchSizE x HDSize -> totWordx
        # wordMatrixReal - totWordZ x HDSize 100 * 1024
        # out = batchSize x totWords x HDSize

        unboundRealExp = tf.reshape(unboundReal, [batchSize, 1, nInput])
        unboundReals   = tf.tile(unboundRealExp, [1, totWords, 1])

        unboundImagExp = tf.reshape(unboundImag, [batchSize, 1, nInput])
        unboundImags   = tf.tile(unboundImagExp, [1, totWords, 1])

        diffReals = tf.sub(unboundReals, allWordsReals)
        diffImags = tf.sub(unboundImags, allWordsImags)
        # batchSize x totWords

        combine = tf.add(tf.square(diffReals), tf.square(diffImags))

        pred = diffNorm = tf.sqrt(tf.reduce_sum(combine, 2))

        cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, targets[i]))

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(targets[i], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return cost, accuracy

cost, accuracy = computeCost(invBindingTF, allWordsTF, targetsTF, memory)
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step * batchSize < trainingIters:
        batchX, batchXRealHD, batchXImagHD, batchIBReal, batchIBImag,  \
                batchTargets = \
                    getBatch(inputData, hdInput, invBindings, targets, batchSize)

        #TODO starting hidden state?
        sess.run(optimizer, feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                    invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                    targetsTF:batchTargets})

        if step % displayStep == 0:
            # Calculate batch loss
            loss, acc = sess.run([cost, accuracy], feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                        invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                        targetsTF:batchTargets})
            print ("Iter %d, Minibatch Loss=%.4f, Accuracy=%.3f" % (step*batchSize, loss, acc))
        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #print ("Testing Accuracy:", \
        #sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
