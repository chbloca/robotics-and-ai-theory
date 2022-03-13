import numpy as np
#import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load data

def load_data():
    # Christen's file directory
    file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
    # Deborah's file directory (customize it and comment the others when using it)
    #file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
    # Ronal's file directory
    #file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
    # Seon's file directory (customize it and comment the others when using it)
    #file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"

    x_test_raw = np.load(file_directory+'X_test_kaggle.npy')
    x_train_raw = np.load(file_directory+'X_train_kaggle.npy')
    # Substitute pandas for loadtxt
    y_train_id = np.loadtxt(file_directory+'groups.csv', delimiter=',', dtype= int, skiprows=0, usecols =(0))
    y_train_idgroup = np.loadtxt(file_directory+'groups.csv', delimiter=',', dtype= int, skiprows=0, usecols =(1))
    y_train_surface = np.loadtxt(file_directory+'groups.csv', delimiter=',', dtype= str, skiprows=0, usecols =(2))
    y_train_raw = np.column_stack((y_train_id, y_train_idgroup, y_train_surface))
    #y_train_raw = pd.read_csv("y_train_final_kaggle.csv", sep=',')
    #group = pd.read_csv("groups.csv", sep=',')
    #group.columns = ["ID", "Group ID", "Surface"]
    #group['Surface'] = group['Surface'].astype('str')
    #group.groupby('Surface').size()
    #group_GroupID = np.array([])
    #group_GroupID = group['Group ID'].values

    return x_train_raw, x_test_raw, y_train_raw

# 2. Create an index of class names

def index_classes(y_train_raw):

    y_train_raw.columns = ["ID", "Label"]
    y_train_raw['Label'] = y_train_raw['Label'].astype('str')

    train_id_array = y_train_raw['ID'].values
    y_train = y_train_raw['Label'].values

    le = LabelEncoder()
    le.fit(y_train)
    y_train_encoded = le.transform(y_train)
    list(y_train_encoded)

    return y_train_encoded

# 3. Split to training and testing

def split(X_train_features, y_train_encoded):

    GSP = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
    x_1, x_2 = GSP.split(X_train_features, y_train_encoded, groups=group_GroupID)
    train_data = X_train_features[x_1[0]]
    test_data = X_train_features[x_1[1]]
    train_labels = y_train_encoded[x_1[0]]
    test_labels = y_train_encoded[x_1[1]]

    #return

# 4. Extract Features

def extract_features_straightforward(x_train_raw):

    X_train_features = np.reshape(x_train_raw, (1703, 1280))

    return X_train_features

def extract_features_average(x_train_raw):
    #Reacontion variable names
    X_train_4 = np.mean(X_train, axis=2)

    Xtrain = X_train_4[x_train[0]]
    Xtest = X_train_4[x_train[1]]

    print("\nUsing Standard Deviation Over Time")
    LDA(Xtrain, Xtest, Ytrain, Ytest)
    print("\n")

    return X_train_features

def extract_features_combination(x_train_raw):

    return X_train_features

# 4. Test LDA

def LDA(x_train, x_test, y_train, y_test):

    model_LDA = LinearDiscriminantAnalysis()
    model_LDA.fit(x_train, y_train)
    result_LDA = model_LDA.predict(x_test)
    score_LDA = accuracy_score(y_test, result_LDA)
    print('LDA Results: ', score_LDA)
    return score_LDA

# 5. Try Different Models

def SVL(x_train, x_test, y_train, y_test):

    model_SVL = SVC(kernel='linear')
    model_SVL.fit(x_train, y_train)
    result_SVL = model_SVL.predict(x_test)
    score_SVL = accuracy_score(y_test, result_SVL)
    print('SVL Results: ', score_SVL)
    return score_SVL

def SVR(x_train, x_test, y_train, y_test):

    model_SVR = SVC(gamma='auto', kernel='rbf')
    model_SVR.fit(x_train, y_train)
    result_SVR = model_SVR.predict(x_test)
    score_SVR = accuracy_score(y_test, result_SVR)
    print('SVR Results: ', score_SVR)
    return score_SVR

def LGR(x_train, x_test, y_train, y_test):

    model_LGR = LogisticRegression(solver='lbfgs', max_iter=500, multi_class='auto')
    model_LGR.fit(x_train, y_train)
    result_LGR = model_LGR.predict(x_test)
    score_LGR = accuracy_score(y_test, result_LGR)
    print('LGR Results: ', score_LGR)
    return score_LGR

def RFC(x_train, x_test, y_train, y_test):

    model_RFC = RandomForestClassifier(n_estimators=500)
    model_RFC.fit(x_train, y_train)
    result_RFC = model_RFC.predict(x_test)
    score_RFC = accuracy_score(y_test, result_RFC)
    print('RFC Results: ', score_RFC)
    return score_RFC


if "__name__" == "__main__":
    # 1. Load data
    x_train_raw, x_train_raw, y_train_raw = load_data()
    # 2. Create an index of class names
    y_train_encoded = index_classes(x_train_raw, x_train_raw, y_train_raw)
    # 3. Split to training and testing
    split(X_train_features, y_train_encoded
    # 4. Extract Features
    X_train_features = extract_features_straightforward(x_train_raw)
    #X_train_features = extract_features_average(x_train_raw)
    #X_train_features = extract_features_combination(x_train_raw)

    # 4. Test LDA
    score_LDA = LDA(train_data, test_data, train_labels, test_labels)
    # 5. Try Different Models
    score_LDA = LDA(Xtrain, Xtest, Ytrain, Ytest)
    score_SVL = SVL(Xtrain, Xtest, Ytrain, Ytest)
    score_SVR = SVR(Xtrain, Xtest, Ytrain, Ytest)
    score_LGR = LGR(Xtrain, Xtest, Ytrain, Ytest)
    score_RFC = RFC(Xtrain, Xtest, Ytrain, Ytest)
