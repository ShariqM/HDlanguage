import numpy as np
import time
import helper as hp
from helper import *
import helper as h
import pdb

def findVec(word, vecMap, hdVecMap, numDim, hdNumDim):
    if word not in vecMap:
        vec = np.random.randn(numDim)
        hdVec = h.generateHDvec(hdNumDim)
        vecMap[word] = (vec, hdVec)
    return vecMap[word]

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

subject = h.generateHDvec(hdNumDim)
color   = h.generateHDvec(hdNumDim)

nQuestions = 2
invBindingsReal = np.zeros((numSamples, nQuestions, hdNumDim))
invBindingsImag = np.zeros((numSamples, nQuestions, hdNumDim))
targetsReal = np.zeros((numSamples, nQuestions, hdNumDim))
targetsImag = np.zeros((numSamples, nQuestions, hdNumDim))

for (i, sentence) in enumerate(sentences):
    words = sentence.split(' ')
    for (j, word) in enumerate(words):
        vec, hdVec = findVec(word, vecMap, hdVecMap, numDim, hdNumDim)
        vecs[i, j, :] = vec
        hdVecsReal[i, j, :] = hdVec.real
        hdVecsImag[i, j, :] = hdVec.imag

    for (j, binding) in enumerate((subject, color)):
        hdInvVec = h.inverseHD(binding)
        invBindingsReal[i, j, :] = hdInvVec.real
        invBindingsImag[i, j, :] = hdInvVec.imag

    # The car is black (want car and black)
    for (j, word) in enumerate((words[1], words[3])):
        vec, hdVec = findVec(word, vecMap, hdVecMap, numDim, hdNumDim)
        targetsReal[i, j, :] = hdVec.real
        targetsImag[i, j, :] = hdVec.imag


# Bindings
numBindings = 10
bindingsReal = np.zeros((numBindings, hdNumDim))
bindingsImag = np.zeros((numBindings, hdNumDim))

for (i, binding) in enumerate((subject, color)):
    bindingsReal[i,:] = binding.real
    bindingsImag[i,:] = binding.imag


def complexMult(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br
i = 0
wi = 3
hr, hi = hdVecsReal[i,wi,:], hdVecsImag[i,wi,:]
br, bi = bindingsReal[i,wi,:], bindingsImag[i,wi,:]
rr, ri = complexMult(hr, hi, br, br)

pdb.set_trace()

pdb.set_trace()
np.savez('data/textVectors', V=vecs, HDVR=hdVecsReal, HDVI=hdVecsImag,
                             BR=bindingsReal, BI=bindingsImag,
                             iBR=invBindingsReal, iBI=invBindingsImag,
                             TR=targetsReal, TI=targetsImag)
