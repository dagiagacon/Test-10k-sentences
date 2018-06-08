import tensorflow as tf
from random import shuffle
import random
x = [i for i in range(1,10)]
shuffle(x)
print(x)
y =random.randint(3,6)
print(y)
num_of_sentences = 500;
max_len_sent = 100
max_number_random = 100
sentences = []
lables = []
for i in range(1,num_of_sentences):
    len_sent = random.randint(1,max_len_sent)
    print(len_sent)
    sent = []
    target = []
    for i in range(1, len_sent+1):
        number = random.randint(1, max_number_random)
        sent.append(number)
        target.append(number+1)
    sentences.append(sent)
    lables.append(target)
#print(sentences)
#print(lables)
print(sentences[1])
file_sent = open("test_sentences.txt","w")
for line in sentences:
     file_sent.write("%s\n" % line)
file_lable = open("test_labels.txt", "w")
for line in lables:
     file_lable.write("%s\n" % line)
