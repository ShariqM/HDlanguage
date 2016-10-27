import numpy as np


## Initialization
N = 10000

currency = np.random.randn(N)
dollar   = np.random.randn(N)
peso     = np.random.randn(N)

country = np.random.randn(N)
usa     = np.random.randn(N)
mexico  = np.random.randn(N)

## Binding
A = country * usa + currency * dollar
B = country * mexico + currency * peso


## What is the dollar of Mexico? (Correct Answer: Peso)
print ("What is the dollar of Mexico? (Peso should have the largest magnitude)")

dollarOfMexico = (A*dollar)*B

for name, hvec in zip(('dollar', 'peso', 'usa', 'mexico'), [dollar, peso, usa, mexico]):
    print ("%s: %.3f" % (name, dollarOfMexico.T @ hvec))


#print (dollar.T @ peso)
#print (dollar.T @ dollar)
