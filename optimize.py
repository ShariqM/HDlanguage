import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from network import Network
import pdb
from helper import complexMultiply

def zeros(shape):
    return np.zeros(shape, dtype=np.float32)

def getBatch(data, dataHD, invBindings, targets, batchSize):
    batchX = zeros((batchSize, data.shape[1], data.shape[2]))


    dataHDReal, dataHDImag = dataHD
    batchXRealHD = zeros((batchSize, dataHD[0].shape[1], dataHD[0].shape[2]))
    batchXImagHD = zeros((batchSize, dataHD[0].shape[1], dataHD[0].shape[2]))

    invBindingsReal, invBindingsImag = invBindings
    batchIBReal = zeros((batchSize, invBindingsReal.shape[1], invBindingsReal.shape[2]))
    batchIBImag = zeros((batchSize, invBindingsReal.shape[1], invBindingsReal.shape[2]))

    targetsReal, targetsImag = targets
    batchTgtReal = zeros((batchSize, targetsReal.shape[1], targetsReal.shape[2]))
    batchTgtImag = zeros((batchSize, targetsReal.shape[1], targetsReal.shape[2]))

    for i in range(batchSize):
        idx = np.random.randint(data.shape[0])

        batchX[i,:,:] = inputData[idx,:,:]
        batchXRealHD[i,:,:] = dataHDReal[idx,:,:]
        batchXImagHD[i,:,:] = dataHDImag[idx,:,:]

        batchIBReal[i,:,:] = invBindingsReal[idx, :, :]
        batchIBImag[i,:,:] = invBindingsImag[idx, :, :]

        batchTgtReal[i,:,:] = targetsReal[idx, :, :]
        batchTgtImag[i,:,:] = targetsImag[idx, :, :]

    return batchX, batchXRealHD, batchXImagHD, batchIBReal, batchIBImag, \
             batchTgtReal, batchTgtImag

allData = np.load('data/textVectors.npz')
inputData = allData['V']
bindingsReal, bindingsImag = allData['BR'], allData['BI']
hdInput = allData['HDVR'], allData['HDVI']
invBindings = allData['iBR'], allData['iBI']
targets = allData['TR'], allData['TI']

nSamples, nWords, nDim = inputData.shape
nHDDim = hdInput[0].shape[2]
nQuestions = invBindings[0].shape[1]
nMemories = nBindings = bindingsReal.shape[0]

# Parameters
learningRate = 1e-2
trainingIters = 1000
batchSize = 2

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

targetRealTF = tf.placeholder("float", [batchSize, nQuestions, nHDInput])
targetImagTF = tf.placeholder("float", [batchSize, nQuestions, nHDInput])
targetTF = (targetRealTF, targetImagTF)

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

def computeCost(invBinding, target, memory):
    invBinding = (stepify(invBinding[0], nHDInput, nQuestions),
                  stepify(invBinding[1], nHDInput, nQuestions))
    target     = (stepify(target[0], nHDInput, nQuestions),
                  stepify(target[1], nHDInput, nQuestions))

    # Hmm?
    wordMatrixRealExp = tf.reshape(wordMatrixReal, [1, totWords, nInput]
    wordMatrixReals   = tf.tile(wordMatrixRealExp, [batchSize, 1, 1])

    wordMatrixImagExp = tf.reshape(wordMatrixImag, [1, totWords, nInput]
    wordMatrixImags   = tf.tile(wordMatrixImagExp, [batchSize, 1, 1])

    invBindingReal, invBindingImag = invBinding
    memoryReal, memoryImag = memory
    targetReal, targetImag = target
    cost = tf.constant(0.0)
    for i in range(nQuestions):
        unboundReal, unboundImag = complexMultiply(invBindingReal[i],
                                                   invBindingImag[i],
                                                   memoryReal, memoryImag)
        # unboundReal - batchSizE x HDSize -> totWordx
        # wordMatrixReal - totWordZ x HDSize 100 * 1024
        # out = batchSize x totWords x HDSize

        unboundRealExp = tf.reshape(unboundedReal, [batchSize, 1, nInput])
        unboundReals   = tf.tile(unboundedRealExp, [1, totWords, 1])

        unboundImagExp = tf.reshape(unboundedImag, [batchSize, 1, nInput])
        unboundImags   = tf.tile(unboundedImagExp, [1, totWords, 1])

        diffReals = tf.subtract(unboundReals, wordMatrixReals)
        diffImags = tf.subtract(unboundImags, wordMatrixImags)
        # batchSize x totWords

        combine = tf.add(tf.square(resultReal), tf.square(resultImag))

        diffNorm = tf.sqrt(tf.reduce_sum(combine, 2))

        cost +== tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, diffNorm))

    return cost

cost = computeCost(invBindingTF, targetTF, memory)
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step * batchSize < trainingIters:
        batchX, batchXRealHD, batchXImagHD, batchIBReal, batchIBImag,  \
                batchTgtReal, batchTgtImag = \
                    getBatch(inputData, hdInput, invBindings, targets, batchSize)

        #TODO starting hidden state?
        sess.run(optimizer, feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                    invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                    targetRealTF:batchTgtReal, targetImagTF:batchTgtImag})

        if step % displayStep == 0:
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                        invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                        targetRealTF:batchTgtReal, targetImagTF:batchTgtImag})
            print ("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #print ("Testing Accuracy:", \
        #sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
