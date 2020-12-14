import numpy as np
import mltools as ml
import sys

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier


class Bagging:
    def __init__(self,learners):
        self.learners = learners
        
    def train(self,X,Y):
        for l in self.learners:
            print(str(type(l)))
            if type(l) in [AdaBoostClassifier, RandomForestClassifier]:
                print("fitting")
                l.fit(X,Y)
                
    
    def predictSoft(self,X):
        Y = np.zeros((X.shape[0], len(self.learners)))
        for i in range(len(self.learners)):
            if type(self.learners[i]) in [AdaBoostClassifier,RandomForestClassifier]:
                Y[:,i] = self.learners[i].predict_proba(X)[:,1]
            else:
                Y[:,i] = self.learners[i].predictSoft(X)[:,1]
        Y = np.mean(Y, axis=1)
        return Y
