import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.utils import to_categorical



# final submission directory
#file_directory ="./"
# Christen's file directory
#file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
# Deborah's file directory (customize it and comment the others when using it)
#file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
# Ronal's file directory
#file_directory = "C:/Users/localadmin/PycharmProjects/Assignment_1/Data/"
# Hoon's file directory (customize it and comment the others when using it)
file_directory = "comp_data/"
# 2. Create an index of class names

def index_classes (y_train_raw):

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train_encoded = le.transform(y_train_raw)
    list(y_train_encoded)
    return y_train_encoded, le

# 1. Load data


def load_data():

    x_test_raw = np.load(file_directory+'X_test_kaggle.npy')
    x_train_raw = np.load(file_directory+'X_train_kaggle.npy')
    #y_train_idgroup = np.loadtxt(file_directory+'groups.csv', delimiter=',', dtype=int, skiprows=0, usecols =(1))
    y_train_raw =np.loadtxt(file_directory + 'y_train_final_kaggle.csv', delimiter=',', dtype=str, skiprows=0, usecols=(1))
    x_test_raw = np.transpose(x_test_raw, (0, 2, 1))
    x_train_raw = np.transpose(x_train_raw,(0, 2, 1))
    y_train_encoded, le = index_classes(y_train_raw)
    y_train_encoded = to_categorical(y_train_encoded)
    return x_train_raw, x_test_raw, y_train_encoded, le


# 3. Split to training and testing


def split_timeseries(X_train_features, X_test_features, y_train_encoded):

    X = X_train_features
    y = y_train_encoded

    tscv = TimeSeriesSplit(n_splits=5)

    for train_index, test_index in tscv.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test,  y_train, y_test


# 4. Extract Features

def extract_features_combination(x_train_raw, x_test_raw):

    x_mean_train = np.mean(x_train_raw, axis=2)
    x_std_train = np.std(x_train_raw, axis=2)
    X_com_train = np.stack((x_mean_train, x_std_train), axis=2)
    X_com_train = np.reshape(X_com_train, (x_train_raw.shape[0], 20))
    x_mean_test = np.mean(x_test_raw, axis=2)
    x_std_test = np.std(x_test_raw, axis=2)
    X_com_test = np.stack((x_mean_test, x_std_test), axis=2)
    X_com_test = np.reshape(X_com_test, (x_test_raw.shape[0], 20))
    return X_com_train, X_com_test

def extract_features_selection(x_train_raw, y_train_encoded, x_test_raw):
    """
    Extract features from dataset

    """
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel

    X_new = [] # Features are stored here
    X_test_new =[]

    x,y,z = x_train_raw.shape
    X_2D = x_train_raw.reshape(x, y*z )

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_2D, y_train_encoded)
    clf.feature_importances_

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X_2D)

    x2,y2,z2 = x_test_raw.shape
    X_test_2D = x_test_raw.reshape(x2, y2*z2 )


def score_analysis(y_test, results):

    score = accuracy_score(y_test, results)
    print('Results: ', score)


def output_file(result_model, lab_enc):

    labels = list(lab_enc.inverse_transform(result_model))
    with open(file_directory+"submission.csv", "w") as fp:
        fp.write("# Id,Surface\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))

"""
LSTM [long short term memory is more powerful than GRU (gated recurrent unit)]
which has to be bidirectional as considers also the future, i.e. considers
the whole sequence at once

Architecture has to be the type of Many-to-one
"""
from sklearn.svm import LinearSVC

def RNN(train_data, val_data, train_labels, val_labels, X_test, losses, reg):

    model = Sequential()
    model.add(LSTM(128, return_sequences = True, input_shape = train_data.shape[1:], bias_regularizer=reg))
    model.add(LSTM(64))
    model.add(Dense(train_labels.shape[1], activation = 'softmax'))

    model.compile(optimizer = 'adagrad', loss = losses, metrics=['accuracy'])
    model.summary()

    model.fit(train_data, train_labels, batch_size = 32, epochs = 25, validation_data = ([val_data, val_labels]))
    score = model.evaluate(val_data, val_labels)
    result_RNN_raw = model.predict(X_test)
    result_RNN = np.argmax(result_RNN_raw, axis = 1)
    train_accuracy = 100*score[1]
    print('Final loss: %.2f' % score[0])
    print('Final accuracy: %.2f %%' % (train_accuracy))

    return result_RNN, result_RNN_raw, train_accuracy

# 0.1 Setting Regularizer (Added 02.03)

from keras.regularizers import L1L2

regularizers = [L1L2(l1=0.0001, l2=0.0)]

# 0.2 Testing for every Keras loss functions (Added 02.03)

loss_fun = ['categorical_crossentropy']

acc_record = []

# 1. Load data
x_train_raw, x_test_raw, y_train_encoded, le = load_data()
train_data, val_data, train_labels, val_labels = split_timeseries(x_train_raw, x_test_raw, y_train_encoded)

for reg in regularizers:
    for losses in loss_fun:
        result_RNN, result_RNN_raw, train_accuracy = RNN(train_data, val_data, train_labels, val_labels, x_test_raw, losses, reg)
        acc_record.append(train_accuracy)

print(acc_record)

# 2. Create an index of class names
   # y_train_encoded, lab_enc = index_classes(y_train_raw)

# 3. Extract Features
#X_train_features, X_test_features = extract_features_stack(x_train_raw, x_test_raw)
#X_train_features, X_test_features  = extract_features_average(x_train_raw)

#X_train_features, X_test_features = extract_features_combination(x_train_raw, x_test_raw)
#D    X_new, X_test_new = extract_features_selection(x_train_raw, y_train_encoded, x_test_raw)

#4. data split using TimeSeriesSplit # 07.02 new addition


#result_model = RNN(train_data, test_data, train_labels)
#score_analysis(test_labels, result_model)

output_file(result_RNN, le)
