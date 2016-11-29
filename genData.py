import numpy as np
import time
import helper as hp
from helper import *
import helper as h
import pdb

def buildVec(word, vecMap, hdVecMap, numDim, hdNumDim):
    if word not in vecMap:
        vec = np.random.randn(numDim)
        hdVec = h.generateHDvec(hdNumDim)
        vecMap[word] = (vec, hdVec)
    return

def buildVecMaps(sentences, vecMap, hdVecMap, numDim, hdNumDim):
    for (i, sentence) in enumerate(sentences):
        words = sentence.split(' ')
        for word in words:
            buildVec(word)

    totWords = len(vecMap)
    allWordsReal = np.zeros((totWords, hdNumDim))
    allWordsImag = np.zeros((totWords, hdNumDim))
    wordMap = {}

    for (i, word, hdVec) in hdVecMap.items():
        allWordsReal[i,:] = hdVec.real
        allWordsImag[i,:] = hdVec.imag
        wordMap[word] = i
    return allWordsReal, allwordsImag, wordMap

fname = 'data/textCorpus.txt'
sentences = loadSentences(fname)

numSamples = len(sentences)
numWords = len(sentences[0].split(' '))
hdNumDim = 2 ** 10
numDim = hdNumDim

vecMap = {}
vecs = np.zeros((numSamples, numWords, numDim))

hdVecMap = {}
hdVecsReal = np.zeros((numSamples, numWords, hdNumDim))
hdVecsImag = np.zeros((numSamples, numWords, hdNumDim))

allWordsReal, allwordsImag, wordMap = buildVecMaps(sentences)

subject = h.generateHDvec(hdNumDim)
color   = h.generateHDvec(hdNumDim)

nQuestions = 2
invBindingsReal = np.zeros((numSamples, nQuestions, hdNumDim))
invBindingsImag = np.zeros((numSamples, nQuestions, hdNumDim))

totWords = len(vecMap)
targets = np.zeros((numSamples, nQuestions, totWords))
#targetsReal = np.zeros((numSamples, nQuestions, hdNumDim))
#targetsImag = np.zeros((numSamples, nQuestions, hdNumDim))

for (i, sentence) in enumerate(sentences):
    words = sentence.split(' ')
    for (j, word) in enumerate(words):
        vec, hdVec = vecMap[word], hdVecMap[word]
        vecs[i, j, :] = vec
        hdVecsReal[i, j, :] = hdVec.real
        hdVecsImag[i, j, :] = hdVec.imag

    for (j, binding) in enumerate((subject, color)):
        hdInvVec = h.inverseHD(binding)
        invBindingsReal[i, j, :] = hdInvVec.real
        invBindingsImag[i, j, :] = hdInvVec.imag

    # "The car is black" (want car and black)
    for (q, word) in enumerate((words[1], words[3])):
        wordIdx = wordMap[word]
        targets[i, q, wordIdx] = 1

# Bindings
numBindings = 10
bindingsReal = np.zeros((numBindings, hdNumDim))
bindingsImag = np.zeros((numBindings, hdNumDim))

for (i, binding) in enumerate((subject, color)):
    bindingsReal[i,:] = binding.real
    bindingsImag[i,:] = binding.imag

np.savez('data/textVectors', V=vecs, HDVR=hdVecsReal, HDVI=hdVecsImag,
                             BR=bindingsReal, BI=bindingsImag,
                             iBR=invBindingsReal, iBI=invBindingsImag,
                             AWR=allWordsReal, AWI=allWordsImag,
                             T=targets)
