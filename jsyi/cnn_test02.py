# cnn lstm model
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, concatenate, Input, Reshape
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import itertools

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

#def load_dataset_cnn():
#    train_x = load_file('../data/UCI HAR Dataset/train/y_train.txt')
#    train_x = list(itertools.chain(*train_x))
#    x1 = train_x
#    x2 = x1
#    x3 = x1
#    X_train = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3})
#    
#    test_x = load_file('../data/UCI HAR Dataset/test/y_test.txt')
#    test_x = list(itertools.chain(*test_x))
#    x1 = test_x
#    x2 = x1
#    x3 = x1
#    X_test = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3})
#    
#    Y_train = load_file('../data/UCI HAR Dataset/train/y_train.txt')
#    Y_test = load_file('../data/UCI HAR Dataset/test/y_test.txt')
#    Y_train = to_categorical(Y_train)
#    Y_test = to_categorical(Y_test)
#
#    return X_train,X_test,Y_train,Y_test

def load_dataset_cnn():
    train_x = load_file('../data/UCI HAR Dataset/train/y_train.txt')
    train_x = list(itertools.chain(*train_x))
    x1 = np.arange(1, 7353)
    x2 = np.arange(1, 7353)
    x3 = np.arange(1, 7353)
    X_train = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3})
    
    test_x = load_file('../data/UCI HAR Dataset/test/y_test.txt')
    test_x = list(itertools.chain(*test_x))
    x1 = np.arange(1, 2948)
    x2 = np.arange(1, 2948)
    x3 = np.arange(1, 2948)
    X_test = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3})

    Y_train = load_file('../data/UCI HAR Dataset/train/y_train.txt')
    Y_test = load_file('../data/UCI HAR Dataset/test/y_test.txt')
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return X_train,X_test,Y_train,Y_test

def get_cnn_model():
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(3,1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    return model

# fit and evaluate a model
def evaluate_model(X_train,X_test,Y_train,Y_test):
    # define model
    cnn_model = get_cnn_model()
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn_model.summary()
    history = cnn_model.fit(X_train.values[:,:,np.newaxis], Y_train, batch_size=256, epochs=100, validation_split=0.2, verbose=1)
    _, accuracy = cnn_model.evaluate(X_test.values[:,:,np.newaxis], Y_test, batch_size=64, verbose=0)
    
    return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=1):
    # load data
    X_train,X_test,Y_train,Y_test = load_dataset_cnn()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train,X_test,Y_train,Y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
 
# run the experiment
run_experiment()