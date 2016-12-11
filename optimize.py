import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from network import Network
import pdb
import matplotlib.pyplot as plt
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
learningRate = 2e-3
trainingIters = 4000
batchSize = 64

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

allWordsRealExp = np.reshape(allWordsReal, [1, totWords, nHDInput])
allWordsReals   = np.tile(allWordsRealExp, [batchSize, 1, 1])
allWordsImagExp = np.reshape(allWordsImag, [1, totWords, nHDInput])
allWordsImags   = np.tile(allWordsImagExp, [batchSize, 1, 1])

allWordsRealTF = tf.constant(allWordsReals.astype(np.float32))
allWordsImagTF = tf.constant(allWordsImag.astype(np.float32))
allWordsTF = (allWordsRealTF, allWordsImagTF)

memoriesRealTF = tf.Variable(tf.zeros([batchSize, nMemories, nHDInput]), trainable=False)
resetMemReal   = memoriesRealTF.assign(tf.zeros([batchSize, nMemories, nHDInput]))
memoriesImagTF = tf.Variable(tf.zeros([batchSize, nMemories, nHDInput]), trainable=False)
resetMemImag   = memoriesRealTF.assign(tf.zeros([batchSize, nMemories, nHDInput]))
memoriesTF     = (memoriesRealTF, memoriesImagTF)

network  = Network(batchSize, nInput, nHDInput, nHidden, nBindings, nMemories, bindingsTF)
hstateTF = network.get_new_state(batchSize) # tfVariable random
hstateVal = np.random.randn(batchSize, nHidden)
resetHState = hstateTF.assign(hstateVal)

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
    accuracies = []

    #scaleParam = tf.Variable(tf.cast(1, tf.float32))
    scaleParam = tf.constant(20.0)
    for i in range(nQuestions):
        unboundReal, unboundImag = complexMultiply(invBindingReal[i],
                                                   invBindingImag[i],
                                                   memoryReal, memoryImag)
        # unboundReal - batchSizE x HDSize -> totWordx
        # wordMatrixReal - totWordZ x HDSize 100 * 1024
        # out = batchSize x totWords x HDSize

        unboundRealExp = tf.reshape(unboundReal, [batchSize, 1, nHDInput])
        unboundReals   = tf.tile(unboundRealExp, [1, totWords, 1])

        unboundImagExp = tf.reshape(unboundImag, [batchSize, 1, nHDInput])
        unboundImags   = tf.tile(unboundImagExp, [1, totWords, 1])

        diffReals = tf.sub(unboundReals, allWordsReals)
        diffImags = tf.sub(unboundImags, allWordsImags)
        # batchSize x totWords

        combine = tf.add(tf.square(diffReals), tf.square(diffImags))

        diffNorm = tf.sqrt(tf.reduce_sum(combine, 2))
        pred = tf.scalar_mul(scaleParam, diffNorm)

        cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, targets[i]))

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(targets[i], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracies.append(accuracy)

    return cost, accuracies

cost, accuracies = computeCost(invBindingTF, allWordsTF, targetsTF, memory)
#optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learningRate).minimize(cost)

batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
  learningRate,       # Base learning rate.
  batch * batchSize,  # Current index into the dataset.
  1000,                # Decay step.
  0.9,               # Decay rate.
  staircase=True)
# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step=batch)
#optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

def clearState(sess):
    sess.run(resetMemReal)
    sess.run(resetMemImag)
    sess.run(resetHState)

lossLog = []
accuracyLog = []
wsLog = []
rsLog = []
with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step * batchSize < trainingIters:
        batchX, batchXRealHD, batchXImagHD, batchIBReal, batchIBImag,  \
                batchTargets = \
                    getBatch(inputData, hdInput, invBindings, targets, batchSize)

        if step == 1 or step % displayStep == 0:
            clearState(sess)
            reading_softmax = network.get_reading_softmax()
            rs = sess.run(reading_softmax, feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                        invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                        targetsTF:batchTargets})
            rsLog.append(np.average(rs, axis=0))
            #print (rs)

            writing_softmaxes = network.get_writing_softmax()
            for writing_softmax in writing_softmaxes:
                clearState(sess)
                ws = sess.run(writing_softmax, feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                            invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                            targetsTF:batchTargets})
                #print ('WS', ws)
            wsLog.append(np.average(ws, axis=0)) # last one




            # Calculate batch loss
            clearState(sess)
            loss = sess.run(cost, feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                        invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                        targetsTF:batchTargets})
            lossLog.append(loss)
            print ("Iter %d, Minibatch Loss=%.4f, Accuracy=[" % (step*batchSize, loss), end="")

            for accuracy in accuracies:
                clearState(sess)
                acc = sess.run(accuracy, feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                            invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                            targetsTF:batchTargets})
                accuracyLog.append(acc)
                print ("%.3f" % acc, end=" ")
            print ("]")

        clearState(sess)
        sess.run(optimizer, feed_dict={xTF:batchX, xHDRealTF:batchXRealHD, xHDImagTF:batchXImagHD, \
                    invBindingRealTF:batchIBReal, invBindingImagTF:batchIBImag, \
                    targetsTF:batchTargets})



        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #print ("Testing Accuracy:", \
        #sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
plt.figure()
plt.title("Accuracy over Time", fontsize=24)
plt.xlabel("Batch Number", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.plot(accuracyLog)
plt.axis([0, len(accuracyLog) - 1, 0, 1.1])

numStates = len(rsLog[0])
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.bar(range(numStates), rsLog[0], width=0.8, tick_label=range(numStates))
plt.title("Read Vector (Start)", fontsize=20)
plt.xlabel("Binding State", fontsize=20)
plt.ylabel("Probability", fontsize=20)
plt.axis([0, numStates, 0, 1])

plt.subplot(2,2,2)
plt.bar(range(numStates), wsLog[0], width=0.8, tick_label=range(numStates))
plt.title("Write Vector (Start)", fontsize=20)
plt.xlabel("Representation State", fontsize=20)
plt.ylabel("Probability", fontsize=20)
plt.axis([0, numStates, 0, 1])

plt.subplot(2,2,3)
plt.bar(range(numStates), rsLog[-1], width=0.8, tick_label=range(numStates))
plt.title("Read Vector (End)", fontsize=20)
plt.xlabel("Binding State", fontsize=20)
plt.ylabel("Probability", fontsize=20)
plt.axis([0, numStates, 0, 1])

plt.subplot(2,2,4)
plt.bar(range(numStates), wsLog[-1], width=0.8, tick_label=range(numStates))
plt.title("Write Vector (End)", fontsize=20)
plt.xlabel("Representation State", fontsize=20)
plt.ylabel("Probability", fontsize=20)
plt.axis([0, numStates, 0, 1])

plt.tight_layout()
plt.show()
