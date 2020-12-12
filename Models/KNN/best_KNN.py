import numpy as np
import mltools as ml
import sys


class KNN_f:
    def __init__(self,X,Y,features, k):
        self.learner = ml.knn.knnClassify()
        self.learner.K = k
        self.selected_feature = np.random.choice(range(X.shape[0]))
        self.learner.train(X[:,self.selected_feature],Y)



class E_knn:
    def __init__(self,learners,features):
        self.classifiers = learners # Allocate space for learners
        features_number = features//10
        self.nbags = nbags
        for i in range(nbags):
            Xi, Yi = ml.bootstrapData(Xtr,Ytr,self.features_number)
            naive_knn = ml.knn.knnClassify()
            naive_knn.train(Xi,Yi)
            naive_knn.K = 16
            self.classifiers[i]=naive_knn
    def predictSoft(self,X):
        Y = np.zeros((X.shape[0], self.nbags))
        for i in range(self.n_boot):
            Y[:,i] = classifiers[i].predict(Xtr)
        Y = np.mean(Y, axis=1)
        return Y
    def soft_error(self,X,Y):
        y_p = self.predictSoft(X)
        return np.mean((y_p-Y)**2)
