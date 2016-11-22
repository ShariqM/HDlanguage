import numpy as np


def mag_angle(wc):
    return np.absolute(wc), np.angle(wc)

def real_imag(mag, angle):
    return mag * np.exp(1j * angle)

def gen(N):
    return real_imag(1, np.random.randn(N))

def inverse(x):
    mag, angle = mag_angle(x)
    return real_imag(mag, -angle)

def norm(r):
    m, a = mag_angle(r)
    return real_imag(1, a)

N = 100000
M = 50
vecs = []
for i in range(M):
    vecs.append(gen(N))
    #print (vecs[i])
    #print (inverse(vecs[i]))
    #print (vecs[i] * inverse(vecs[i]))
    #print (vecs[i] * gen(N))
    #break


brown = vecs[0]
color = vecs[1]

carl  = vecs[2]
subject = vecs[3]


yellow = vecs[4]
green = vecs[5]
blue = vecs[6]

action = vecs[7]
jump = vecs[8]

father = vecs[9]
james = vecs[10]

r = norm(brown * color)
print (np.linalg.norm(r * inverse(color) - brown))
print (np.linalg.norm(r * inverse(color) - yellow))
print (np.linalg.norm(r * inverse(color) - green))
print (np.linalg.norm(r * inverse(color) - blue))

print ('exp2')
r = norm(brown * color + carl * subject)
print (np.linalg.norm(r * inverse(color) - brown))
print (np.linalg.norm(r * inverse(color) - yellow))
print (np.linalg.norm(r * inverse(color) - green))
print (np.linalg.norm(r * inverse(color) - blue))

print ('exp3')
r = norm(brown * color + carl * subject + jump * action)
print (np.linalg.norm(r * inverse(color) - brown))
print (np.linalg.norm(r * inverse(color) - yellow))
print (np.linalg.norm(r * inverse(color) - green))
print (np.linalg.norm(r * inverse(color) - blue))

print ('exp4')
r = norm(brown * color + carl * subject + jump * action + father * james)
print (np.linalg.norm(r * inverse(color) - brown))
print (np.linalg.norm(r * inverse(color) - yellow))
print (np.linalg.norm(r * inverse(color) - green))
print (np.linalg.norm(r * inverse(color) - blue))





#for i in range(M):
    #for j in range(M):
        #print (i,j, complexMultiply(vecs[i], vecs[j]))

