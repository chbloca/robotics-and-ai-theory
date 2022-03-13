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

# final submission directory
#file_directory ="./"
# Christen's file directory
file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
# Deborah's file directory (customize it and comment the others when using it)
#file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"
# Ronal's file directory
#file_directory = "C:/Users/localadmin/PycharmProjects/Assignment_1/Data/"

# Seon's file directory (customize it and comment the others when using it)
#file_directory = "C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Competition ML & PR/robotsurface/"


# 1. Load data


def load_data():

    x_test_raw = np.load(file_directory+'X_test_kaggle.npy')
    x_train_raw = np.load(file_directory+'X_train_kaggle.npy')
    # Substitute pandas for loadtxt
    y_train_idgroup = np.loadtxt(file_directory+'groups.csv', delimiter=',', dtype= int, skiprows=0, usecols =(1))
    y_train_raw =np.loadtxt(file_directory + 'y_train_final_kaggle.csv', delimiter=',', dtype=str, skiprows=0, usecols=(1))
    return x_train_raw, x_test_raw, y_train_raw, y_train_idgroup

# 2. Create an index of class names


def index_classes (y_train_raw):

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train_encoded = le.transform(y_train_raw)
    list(y_train_encoded)
    return y_train_encoded, le

# 3. Split to training and testing


def split_tt(X_train_features, y_train_encoded, y_train_idgroup):

    GSP = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
    x_1, x_2 = GSP.split(X_train_features, y_train_encoded, groups=y_train_idgroup)
    train_data = X_train_features[x_1[0]]
    test_data = X_train_features[x_1[1]]
    train_labels = y_train_encoded[x_1[0]]
    test_labels = y_train_encoded[x_1[1]]
    return train_data, train_labels, test_data, test_labels

    
def split_timeseries(X_train_features, X_test_features, y_train_encoded):

    X = X_train_features
    y = y_train_encoded
 
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test,  y_train, y_test 
    

# 4. Extract Features


def extract_features_stack(x_train_raw, x_test_raw):


    X_train_features = np.reshape(x_train_raw, (x_train_raw.shape[0], 1280))
    X_test_features = np.reshape(x_test_raw, (x_test_raw.shape[0], 1280))
    return X_train_features, X_test_features


def extract_features_average(x_train_raw):


    X_train_av = np.mean(x_train_raw, axis=2)
    X_test_av = np.mean(x_test_raw, axis=2)
    return X_train_av, X_test_av


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
 
    X_test_new = model.transform(X_test_2D)
    
  
    return np.array(X_new), np.array(X_test_new)

    

# 4. Test LDA


def LDA(x_train, x_test, y_train):

    model_LDA = LinearDiscriminantAnalysis()
    model_LDA.fit(x_train, y_train)
    result_LDA = model_LDA.predict(x_test)
    print("LDA model executed \n")
    return result_LDA

# 5. Try Different Models


def SVL(x_train, x_test, y_train):

    model_SVL = SVC(kernel='linear')
    model_SVL.fit(x_train, y_train)
    result_SVL = model_SVL.predict(x_test)
    print("SVL model executed \n")
    return result_SVL


def SVR(x_train, x_test, y_train):

    model_SVR = SVC(gamma='auto', kernel='rbf')
    model_SVR.fit(x_train, y_train)
    result_SVR = model_SVR.predict(x_test)
    print("SVR model executed \n")
    return result_SVR


def LGR(x_train, x_test, y_train):

    model_LGR = LogisticRegression(solver='lbfgs', max_iter=500, multi_class='auto')
    model_LGR.fit(x_train, y_train)
    result_LGR = model_LGR.predict(x_test)
    print("LGR model executed \n")
    return result_LGR


def RFC(x_train, x_test, y_train):

    model_RFC = RandomForestClassifier(n_estimators=500)
    model_RFC.fit(x_train, y_train)
    result_RFC = model_RFC.predict(x_test)
    print("RFC model executed \n")
    return result_RFC

def GradientBoosting(x_train, x_test, y_train):
    model_GB =  GradientBoostingClassifier()
    model_GB.fit(x_train, y_train)
    result_GB = model_GB.predict(x_test)
    print("GredientBoosting model executed \n")
    return result_GB

def score_analysis(y_test, results):

    score = accuracy_score(y_test, results)
    print('Results: ', score)


def output_file(result_model, lab_enc):

    labels = list(lab_enc.inverse_transform(result_model))
    with open(file_directory+"submission.csv", "w") as fp:
        fp.write("# Id,Surface\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
            
if __name__ == "__main__":

    # 1. Load data
    x_train_raw, x_test_raw, y_train_raw, y_train_idgroup = load_data()
    
    # 2. Create an index of class names
    y_train_encoded, lab_enc = index_classes(y_train_raw)
    
    # 3. Extract Features
    X_train_features, X_test_features = extract_features_stack(x_train_raw, x_test_raw)
    #X_train_features, X_test_features  = extract_features_average(x_train_raw)
    
    #X_train_features, X_test_features = extract_features_combination(x_train_raw, x_test_raw)
    #D    X_new, X_test_new = extract_features_selection(x_train_raw, y_train_encoded, x_test_raw) 

    #4. data split using TimeSeriesSplit # 07.02 new addition 
 
    train_data, test_data, train_labels, test_labels = split_timeseries(X_train_features, X_test_features, y_train_encoded)

   

    #D X_new data split with two groups of train and test  
    #    train_data, test_data, train_labels, test_labels = train_test_split(X_new, y_train_encoded, test_size=0.20) 
    
    
    # 3. Split to training and testing
    #D   train_data, train_labels, test_data, test_labels = split_tt(X_train_features, y_train_encoded, y_train_idgroup)
    #D    train_data, train_labels, test_data, test_labels = split_tt(X_new, y_train_encoded, y_train_idgroup) #MK
 
    #X_train_features = extract_features_average(x_train_raw)
    #X_train_features = extract_features_combination(x_train_raw)
    # 4. Test LDA
    result_model = LDA(train_data, test_data, train_labels)
    # 5. Try Different Models
    #    result_model = SVL(train_data, test_data, train_labels)
    #    score_analysis(test_labels, result_model)
    
    #result_model = GradientBoosting(train_data, test_data, train_labels)
    score_analysis(test_labels, result_model)

    #result_model = SVR(train_data, test_data, train_labels)
    #result_model = LGR(train_data, test_data, train_labels)
    #result_model = RFC(train_data, test_data, train_labels)
    #D    score_analysis(test_labels, result_model)
    
    #6. test file things
    #D    final_result_model = SVL(train_data, X_test_features, train_labels)
    #final_result_model = RFC(train_data, X_test_features, train_labels)
    
    #output_file(final_result_model, lab_enc)
