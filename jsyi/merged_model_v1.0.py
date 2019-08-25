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

#def load_dataset_cnn():
#    train_x = load_file('../data/UCI HAR Dataset/train/y_train.txt')
#    train_x = list(itertools.chain(*train_x))
#    x1 = train_x
#    x2 = x1
#    x3 = x1
#    x4 = x1
#    X_train = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'x4':x4})
#    
#    test_x = load_file('../data/UCI HAR Dataset/test/y_test.txt')
#    test_x = list(itertools.chain(*test_x))
#    x1 = test_x
#    x2 = x1
#    x3 = x1
#    x4 = x1
#    X_test = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'x4':x4})
#    
#    return X_train,X_test

def load_dataset_cnn():
    train_x = load_file('../data/UCI HAR Dataset/train/y_train.txt')
    train_x = list(itertools.chain(*train_x))
    x1 = np.arange(1, 7353)
    x2 = np.arange(1, 7353)
    x3 = np.arange(1, 7353)
    x4 = np.arange(1, 7353)
    X_train = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'x4':x4})
    
    test_x = load_file('../data/UCI HAR Dataset/test/y_test.txt')
    test_x = list(itertools.chain(*test_x))
    x1 = np.arange(1, 2948)
    x2 = np.arange(1, 2948)
    x3 = np.arange(1, 2948)
    x4 = np.arange(1, 2948)
    X_test = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'x4':x4})

    return X_train,X_test

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
 
# load a list of files and return as a 3d numpy array
def load_group_lstm(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded
 
# load a dataset group, such as train or test
def load_dataset_group_lstm(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group_lstm(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset_lstm(prefix='../data/'):
	# load all train
	trainX, trainy = load_dataset_group_lstm('train', prefix + 'UCI HAR Dataset/')
	print('trainX.shape, trainy.shape: ',trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group_lstm('test', prefix + 'UCI HAR Dataset/')
	print('testX.shape, testy.shape: ',testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print('trainX.shape, trainy.shape, testX.shape, testy.shape: ',trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, testX, trainy, testy

def get_cnn_model():
    inp = Input(shape=(4,1))
    model = Conv1D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu')(inp)
    model = MaxPooling1D(pool_size=(2), strides=(1))(model)
    model = Dropout(0.3)(model)
    model = Flatten()(model)
    model = Dense(100, activation='relu')(model)

    return inp, model

def get_lstm_model(n_length,n_features, n_outputs):
    inp = Input(shape=(None,n_length,n_features))
    model = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'))(inp)
    model = TimeDistributed(Dropout(0.5))(model)
    model = TimeDistributed(MaxPooling1D(pool_size=2))(model)
    model = TimeDistributed(Flatten())(model)
    model = LSTM(100)(model)
    model = Dropout(0.5)(model)
    model = Dense(100, activation='relu')(model)
#    model = Dense(n_outputs, activation='softmax')(model)
    
    return inp, model

def get_merged_model(cnn_inp, lstm_inp, cnn_model, lstm_model, n_outputs):
    x = concatenate([cnn_model, lstm_model])
    x = Dense(32, kernel_initializer='normal', activation='relu', name='merged_dense_1')(x)
    x = Dropout(0.5, name='x_drop_1')(x)
    x = Dense(8, kernel_initializer='normal', activation='relu', name='merged_dense_2')(x)
    x = Dropout(0.5, name='x_drop_2')(x)
    fx = Dense(n_outputs, activation='softmax', name='fx')(x)

    model = Model(inputs=[cnn_inp, lstm_inp], output=fx)

    return model
# fit and evaluate a model
def evaluate_model(cx_train, cx_test, lx_train, lx_test, y_train, y_test):
    # define model
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = lx_train.shape[1], lx_train.shape[2], y_train.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 32
    lx_train = lx_train.reshape((lx_train.shape[0], n_steps, n_length, n_features))
    lx_test = lx_test.reshape((lx_test.shape[0], n_steps, n_length, n_features))
    # define model
    cnn_inp, cnn_model = get_cnn_model()
    lstm_inp, lstm_model = get_lstm_model(n_length,n_features,n_outputs)
    merged_model = get_merged_model(cnn_inp, lstm_inp, cnn_model, lstm_model, n_outputs)
    merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    merged_model.summary()
    # fit network
    history = merged_model.fit([cx_train.values[:,:,np.newaxis], lx_train], y_train, batch_size=256, epochs=500, validation_split=0.2, verbose=1)
    # evaluate model
    _, accuracy = merged_model.evaluate([cx_test.values[:,:,np.newaxis], lx_test], y_test, batch_size=batch_size, verbose=0)
    
#    return 1
    return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=1):
    # load data
    cx_train, cx_test = load_dataset_cnn()
    lx_train, lx_test, y_train, y_test = load_dataset_lstm()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(cx_train, cx_test, lx_train, lx_test, y_train, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
 
# run the experiment
run_experiment()