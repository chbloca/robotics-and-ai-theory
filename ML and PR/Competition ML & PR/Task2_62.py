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
from keras.layers import Dense, Dropout, Embedding, LSTM, BatchNormalization
from keras.utils import to_categorical
from keras import regularizers


import matplotlib.pyplot as plt


# final submission directory
file_directory ="./"
# Christen's file directory
#file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
# Deborah's file directory (customize it and comment the others when using it)
#file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
# Ronal's file directory
#file_directory = "C:/Users/localadmin/PycharmProjects/Assignment_1/Data/"
# Seon's file directory (customize it and comment the others when using it)
#file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"

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
    y_train_idgroup = np.loadtxt(file_directory+'groups.csv', delimiter=',', dtype=int, skiprows=0, usecols =(1))
    y_train_raw =np.loadtxt(file_directory + 'y_train_final_kaggle.csv', delimiter=',', dtype=str, skiprows=0, usecols=(1))
    x_test_raw = np.transpose(x_test_raw, (0, 2, 1))
    x_train_raw = np.transpose(x_train_raw,(0, 2, 1))
    y_train_encoded, le = index_classes(y_train_raw)
    y_train_1D =  y_train_encoded
    y_train_encoded = to_categorical(y_train_encoded)
    return x_train_raw, x_test_raw, y_train_encoded, le, y_train_1D


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

def feature_selectKBest(x_train_raw, y_train_1D, x_test_raw):
    """
    Extract features from dataset using selectKBest
 
    """
    from sklearn.feature_selection import SelectKBest, f_regression 

    x,y,z = x_train_raw.shape
    X_2D = x_train_raw.reshape(x, y*z )
    

    X_selected = SelectKBest(f_regression, k=40).fit_transform(X_2D, y_train_1D)
    model = SelectKBest(f_regression, k=40)
    X_selected =  model.fit_transform(X_2D, y_train_1D)
#        X_selected, y_selected = SelectKBest(f_regression, k=40).fit_transform(X_2D, y_train_1D)
    
    x,y = X_selected.shape
    X_selected_3D = X_selected.reshape(x,y,1)
    y_selected_categorical = to_categorical(y_train_1D)
    
    x2,y2,z2 = x_test_raw.shape
    X_test_2D = x_test_raw.reshape(x2, y2*z2 )
 
    X_test_new = model.transform(X_test_2D)
    
    x,y = X_test_new.shape
    X_test_3D = X_test_new.reshape(x,y,1)

    return X_selected_3D, y_selected_categorical, X_test_3D



def feature_selectVariance(x_train_raw, y_train_1D, x_test_raw):
    """
    Extract features from dataset using VarianceThreshold
 
    """
 
    from sklearn.feature_selection import VarianceThreshold
 
 
    x,y,z = x_train_raw.shape
    X_2D = x_train_raw.reshape(x, y*z )
    
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_selected = sel.fit_transform(X_2D, y_train_1D)
 
   
    x,y = X_selected.shape
    X_selected_3D = X_selected.reshape(x,y,1)
    y_selected_categorical = to_categorical(y_train_1D)
    
    x2,y2,z2 = x_test_raw.shape
    X_test_2D = x_test_raw.reshape(x2, y2*z2 )
 
    X_test_new = sel.fit_transform(X_test_2D)
    
    x,y = X_test_new.shape
    X_test_3D = X_test_new.reshape(x,y,1)

    return X_selected_3D, y_selected_categorical, X_test_3D

    
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
# Use feature selection as input?

def RNN(train_data, val_data, train_labels, val_labels, X_test):
    
    model = Sequential()
    model.add(LSTM(64, return_sequences = True, input_shape = train_data.shape[1:], kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l1(0.001)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(LSTM(64, kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l1(0.001)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(train_labels.shape[1], activation = 'softmax', kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l1(0.001)))  
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train_data, train_labels, batch_size = 32, epochs = 20, validation_data = ([val_data, val_labels]))
    score = model.evaluate(val_data, val_labels)
    result_RNN_raw = model.predict(X_test)
    result_RNN = np.argmax(result_RNN_raw, axis = 1)
    print('Final loss: %.2f' % score[0])
    print('Final accuracy: %.2f %%' % (100*score[1]))

    return result_RNN, result_RNN_raw

def RNN_p(train_data, val_data, train_labels, val_labels, X_test, l1_p, epoch_p, batch_p):
    
    model = Sequential()
    model.add(LSTM(100, return_sequences = True, input_shape = train_data.shape[1:], kernel_regularizer=regularizers.l1(l=l1_p), activity_regularizer=regularizers.l1(l=l1_p)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(LSTM(300, kernel_regularizer=regularizers.l1(l=l1_p), activity_regularizer=regularizers.l1(l=l1_p)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(train_labels.shape[1], activation = 'softmax', kernel_regularizer=regularizers.l1(l=l1_p), activity_regularizer=regularizers.l1(l=l1_p)))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_data, train_labels, batch_size = batch_p, epochs = epoch_p, validation_data = ([val_data, val_labels]))
    score = model.evaluate(val_data, val_labels)
    result_RNN_raw = model.predict(X_test)
    result_RNN = np.argmax(result_RNN_raw, axis = 1)
    print('Final loss: %.2f' % score[0])
    print('Final accuracy: %.2f %%' % (100*score[1]))
    
    
    return result_RNN, result_RNN_raw, model, history

def param_test (l1_p, epoch_p, batch_p):

    l1_range = np.logspace(-5,0,6)
    
    for l1_p in l1_range:
        print('Start of loop =================================================l1=: ', l1_p)
        result_RNN, result_RNN_raw = RNN_p(train_data, val_data, train_labels, val_labels, x_test_raw, l1_p, epoch_p, batch_p)
        print('End of loop =================================================l1=: ', l1_p)
            
    return

def model_visualization(history):
    #visualization

    # Plot training & validation accuracy values
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return



# 1. Load data
x_train_raw, x_test_raw, y_train_encoded, le, y_train_1D = load_data()

#feature selection using selectKBest
#X_train_selected, y_selected, x_test_selected =feature_selectKBest(x_train_raw, y_train_1D, x_test_raw)

#feature selection using selectVariance
X_train_selected, y_selected, x_test_selected = feature_selectVariance(x_train_raw, y_train_1D, x_test_raw)


train_data, val_data, train_labels, val_labels = split_timeseries(X_train_selected, x_test_selected, y_selected)

#train_data, val_data, train_labels, val_labels = split_timeseries(x_train_raw, x_test_raw, y_train_encoded)
#result_RNN, result_RNN_raw = RNN(train_data, val_data, train_labels, val_labels, x_test_raw)


#Raw data test with more layer parameters
train_data, val_data, train_labels, val_labels = split_timeseries(x_train_raw, x_test_raw, y_train_encoded)


l1_p = 1e-04
batch_p=50
epoch_p = 35
result_RNN, result_RNN_raw, model, history = RNN_p(train_data, val_data, train_labels, val_labels, x_test_raw,l1_p, epoch_p, batch_p)
model_visualization(history)

#########
#l1_p = 1e-05
#batch_p=50
#epoch_p = 35
#result_RNN, result_RNN_raw, model, history = RNN_p(train_data, val_data, train_labels, val_labels, x_test_selected, l1_p, epoch_p, batch_p)
#
#model_visualization(history)


## Tuning the parameters 
#l1_range = np.logspace(-5,0,6)
#batch_p=50
#epoch_p = 35
#param_test (l1_p, epoch_p, batch_p)
#
# 

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
