from NNC import NearestNeighbor
import numpy as np
from cifar_utils import load_CIFAR10

Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')

Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

'''in order to test, can set test/training data number by following setting'''
Xtr_rows = Xtr_rows[0:10000]
Xte_rows = Xte_rows[0:1000]

nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)

# print(Yte_predict, Yte)
count = 0
for i in range(len(Yte_predict)):
    if Yte_predict[i] == Yte[i]:
        count += 1
res = count/len(Yte_predict)
print('accuracy: %f' % res)
