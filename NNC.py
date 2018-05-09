import numpy as np


class NearestNeighbor(object):

    def __index__(self):
        pass

    def train(self, X, y):
        '''X is N*D where each row is an example, Y is one-D of size N'''
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        '''X is X*D where each row is an example we wish to predict label for '''
        num_test = X.shape[0]

        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            distance = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
            min_index = np.argmin(distance)
            Ypred[i] = self.ytr[min_index]
            print(i, num_test)

        return Ypred

