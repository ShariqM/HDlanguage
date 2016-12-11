import numpy as np
import time
import helper as hp
from helper import *
import helper as h
import pdb

def buildVec(word, vecMap, numDim, hdNumDim):
    if word not in vecMap:
        vec = np.random.randn(numDim)
        hdVec = h.generateHDvec(hdNumDim)
        vecMap[word] = (vec, hdVec)
    return

def buildVecMaps(sentences, vecMap,numDim, hdNumDim):
    for (i, sentence) in enumerate(sentences):
        words = sentence.split(' ')
        for word in words:
            buildVec(word, vecMap, numDim, hdNumDim)

    totWords = len(vecMap)
    allWordsReal = np.zeros((totWords, hdNumDim))
    allWordsImag = np.zeros((totWords, hdNumDim))
    wordMap = {}

    for (i, (word, vecs)) in enumerate(vecMap.items()):
        allWordsReal[i,:] = vecs[1].real
        allWordsImag[i,:] = vecs[1].imag
        wordMap[word] = i
    return allWordsReal, allWordsImag, wordMap

fname = 'data/textCorpus.txt'
sentences = loadSentences(fname)
print (sentences)

numSamples = len(sentences)
numWords = len(sentences[0].split(' '))
hdNumDim = 2 ** 10
numDim = 2 ** 8

vecMap = {}
vecs = np.zeros((numSamples, numWords, numDim))
hdVecsReal = np.zeros((numSamples, numWords, hdNumDim))
hdVecsImag = np.zeros((numSamples, numWords, hdNumDim))

allWordsReal, allWordsImag, wordMap = \
        buildVecMaps(sentences, vecMap, numDim, hdNumDim)

# XXX HACK TEST HACK
sentences = [sentences[i] for i in range(16)]
#sentences[0] = "brown"
print (len(vecMap))
print (sentences)

subject = h.generateHDvec(hdNumDim)
color   = h.generateHDvec(hdNumDim)

nQuestions = 1
invBindingsReal = np.zeros((numSamples, nQuestions, hdNumDim))
invBindingsImag = np.zeros((numSamples, nQuestions, hdNumDim))

totWords = len(vecMap)
targets = np.zeros((numSamples, nQuestions, totWords))
#targetsReal = np.zeros((numSamples, nQuestions, hdNumDim))
#targetsImag = np.zeros((numSamples, nQuestions, hdNumDim))

for (i, sentence) in enumerate(sentences):
    words = sentence.split(' ')
    for (j, word) in enumerate(words):
        vec, hdVec = vecMap[word]
        vecs[i, j, :] = vec
        hdVecsReal[i, j, :] = hdVec.real
        hdVecsImag[i, j, :] = hdVec.imag

    #for (j, binding) in enumerate((subject, color)):
    for (j, binding) in enumerate((color,)):
        hdInvVec = h.inverseHD(binding)
        invBindingsReal[i, j, :] = hdInvVec.real
        invBindingsImag[i, j, :] = hdInvVec.imag

    # "The car is black" (want car and black)
    #for (q, word) in enumerate((words[1], words[3])):
    for (q, word) in enumerate((words[0],)):
        wordIdx = wordMap[word]
        targets[i, q, wordIdx] = 1

# Bindings
numBindings = 2
bindingsReal = np.random.randn(numBindings, hdNumDim)
bindingsImag = np.random.randn(numBindings, hdNumDim)

#for (i, binding) in enumerate((subject, color)):
for (i, binding) in enumerate((color,)):
    bindingsReal[i,:] = binding.real
    bindingsImag[i,:] = binding.imag

np.savez('data/textVectors', V=vecs, HDVR=hdVecsReal, HDVI=hdVecsImag,
                             BR=bindingsReal, BI=bindingsImag,
                             iBR=invBindingsReal, iBI=invBindingsImag,
                             AWR=allWordsReal, AWI=allWordsImag,
                             T=targets)
