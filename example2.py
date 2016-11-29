import numpy as np
import pdb


def mag_angle(c):
    return np.absolute(c), np.angle(c)

def real_imag(mag, angle):
    return mag * np.exp(1j * angle)

def gen(N):
    return real_imag(1, np.random.uniform(0, 2*np.pi, N))
    #return real_imag(1, np.random.randn(N))

def inverse(x):
    mag, angle = mag_angle(x)
    return real_imag(mag, -angle)

def norm(r):
    m, a = mag_angle(r)
    return real_imag(1, a)

N = 2 ** 10
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

def func1(r, color, var):
    #return np.sum((r * inverse(color) * inverse(var)).imag)
    vec = (r * inverse(color) * inverse(var))
    mag, angle = mag_angle(vec)
    #pdb.set_trace()
    #vecNew = np.pi - np.abs(angle)
    return np.sum(np.abs(angle))

def test1(name, r):
    print ("%s.1" % name)
    for var in (brown, yellow, green, blue):
        print (func1(r, color, var))

def test2(name, r):
    print ("%s.2" % name)
    print (np.linalg.norm(r * inverse(color) - brown))
    print (np.linalg.norm(r * inverse(color) - yellow))
    print (np.linalg.norm(r * inverse(color) - green))
    print (np.linalg.norm(r * inverse(color) - blue))

def test(name, r):
    test1(name, r)
    test2(name, r)

r = norm(brown * color)
test('A', r)

r = norm(brown * color + carl * subject)
test('B', r)

r = norm(brown * color + carl * subject + jump * action)
test('C', r)

r = norm(brown * color + carl * subject + jump * action + father * james)
test('D', r)

#print ('absolute')
#print (np.absolute((r * inverse(color)).T @ brown))
#print (np.absolute((r * inverse(color)).T @ yellow))
#print (np.absolute((r * inverse(color)).T @ green))
#print (np.absolute((r * inverse(color)).T @ blue))

#for i in range(M):
    #for j in range(M):
        #print (i,j, complexMultiply(vecs[i], vecs[j]))

