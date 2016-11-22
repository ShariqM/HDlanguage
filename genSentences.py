import numpy as np
import random

#names = ["Amanda", "John", "Charles", "Michael"]

subjects = ["dog", "cat", "wolf", "car"]
colors = ["brown", "yellow", "black", "white"]

sentences = []

for subject in subjects:
    for color in colors:
        sentence = "The " + subject + " is " + color
        sentences.append(sentence)


random.shuffle(sentences)
f = open('data/textCorpus.txt', 'w')
for sentence in sentences:
    f.write("%s\n" % sentence)
