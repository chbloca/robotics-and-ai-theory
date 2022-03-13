import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
digits = load_digits()

def Load_Data():

     plt.gray()
     plt.imshow(digits.images[0])
     plt.show()
     data=digits['data']
     target=digits['target']
     images=digits['images']
     print(digits.keys())
     x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
     return x_train, x_test, y_train, y_test

def KNN(x_train, x_test, y_train, y_test):

    model_KNN = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    model_KNN.fit(x_train, y_train)
    result_KNN = model_KNN.predict(x_test)
    score_KNN = accuracy_score(y_test, result_KNN)
    print('KNN Results: ', score_KNN)
    return score_KNN

def LDA(x_train, x_test, y_train, y_test):

    model_LDA = LinearDiscriminantAnalysis()
    model_LDA.fit(x_train, y_train)
    result_LDA = model_LDA.predict(x_test)
    score_LDA = accuracy_score(y_test, result_LDA)
    print('LDA Results: ', score_LDA)
    return score_LDA

def SVC_model(x_train, x_test, y_train, y_test):

    model_SVC = SVC(gamma='auto')
    model_SVC.fit(x_train, y_train)
    result_SVC = model_SVC.predict(x_test)
    score_SVC = accuracy_score(y_test, result_SVC)
    print('SVC Results: ', score_SVC)
    return score_SVC

def LR(x_train, x_test, y_train, y_test):

    model_LR = LogisticRegression()
    model_LR.fit(x_train, y_train)
    result_LR = model_LR.predict(x_test)
    score_LR = accuracy_score(y_test, result_LR)
    print('LR Results: ', score_LR)
    return score_LR

if __name__ == "__main__":

    #3 Load dataset split to training and tesitng
    x_train, x_test, y_train, y_test = Load_Data()
    #3 Classifiers
    KNN_R = KNN(x_train, x_test, y_train, y_test)
    LDA_R = LDA(x_train, x_test, y_train, y_test)
    SVC_R = SVC_model(x_train, x_test, y_train, y_test)
    LR_R = LR(x_train, x_test, y_train, y_test)
    #3 Best Classifier
    Names= ['KNN', 'LDA', 'SVC', 'LR']
    Results = [KNN_R, LDA_R, SVC_R, LR_R]
    print('Best result: ', Names[np.argmax(Results)], max(Results))

