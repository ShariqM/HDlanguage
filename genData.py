import numpy as np
import time

def loadSentences(fname):
    sentences = []
    f = open(fname, 'r')
    x = f.readline()
    while x != '':
        sentences.append(x)
        x = f.readline()
    return sentences

def findVec(vecMap, word):
    if word not in vecMap:
        vecMap[word] = np.random.randn(numDim)
    return vecMap[word]

fname = 'data/textCorpus.txt'
sentences = loadSentences(fname)

numSamples = len(sentences)
numWords = len(sentences[0].split(' '))
numDim = 2 ** 10

vecMap = {}
data = np.zeros((numSamples, numWords, numDim))

for (i, sentence) in enumerate(sentences):
    words = sentence.split(' ')
    for (j, word) in enumerate(words):
        data[i, j, :] = findVec(vecMap, word)

np.savez('data/textVectors', X=data)

# Bindings
color = np.random.randn(numDim)
name =  np.random.randn(numDim)
numBindings = 10
bindings = np.zeros((numBindings, numDim))
np.savez('data/textBindings', X=bindings)
