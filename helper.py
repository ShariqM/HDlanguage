import numpy as np
import tensorflow as tf

def loadSentences(fname):
    sentences = []
    f = open(fname, 'r')
    x = f.readline()
    while x != '':
        sentences.append(x)
        x = f.readline()
    return sentences

def complexMultiply(AReal, AImag, BReal, BImag):
    outReal = tf.mul(AReal, BReal) - tf.mul(AImag, BImag)
    outImag = tf.mul(AReal, BImag) + tf.mul(AImag, BReal)
    return (outReal, outImag)

def complexMultiplyMany(AReal, AImag, BReal, BImag):
    outReal = tf.matmul(AReal, BReal) - tf.matmul(AImag, BImag)
    outImag = tf.matul(AReal, BImag) + tf.matmul(AImag, BReal)
    return (outReal, outImag)

def mag_angle(wc):
    return np.absolute(wc), np.angle(wc)

def real_imag(mag, angle):
    return mag * np.exp(1j * angle)

def generateHDvec(N):
    hdVec =real_imag(1, np.random.uniform(0, 2*np.pi, N))
    hdVec[np.random.choice(range(1024))
    return real_imag(1, np.random.uniform(0, 2*np.pi, N))

def inverseHD(x):
    mag, angle = mag_angle(x)
    return real_imag(mag, -angle)

def norm(r):
    m, a = mag_angle(r)
    return real_imag(1, a)
