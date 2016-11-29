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

    batchTargets = zeros((batchSize, targets.shape[1], targets.shape[2]))

    for i in range(batchSize):
        idx = np.random.randint(data.shape[0])

        batchX[i,:,:] = data[idx,:,:]
        batchXRealHD[i,:,:] = dataHDReal[idx,:,:]
        batchXImagHD[i,:,:] = dataHDImag[idx,:,:]

        batchIBReal[i,:,:] = invBindingsReal[idx, :, :]
        batchIBImag[i,:,:] = invBindingsImag[idx, :, :]

        batchTargets[i,:,:] = targets[idx, :, :]

    return batchX, batchXRealHD, batchXImagHD, batchIBReal, batchIBImag, \
             batchTargets

def mag_angle(wc):
    return np.absolute(wc), np.angle(wc)

def real_imag(mag, angle):
    return mag * np.exp(1j * angle)

def generateHDvec(N):
    #hdVec =real_imag(1, np.random.uniform(0, 2*np.pi, N))
    #hdVec[np.random.choice(range(1024)) XXX Zero out some??
    return real_imag(1, np.random.uniform(0, 2*np.pi, N))

def inverseHD(x):
    mag, angle = mag_angle(x)
    return real_imag(mag, -angle)

def norm(r):
    m, a = mag_angle(r)
    return real_imag(1, a)


