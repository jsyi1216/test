#!/share/anaconda2/bin/python
"""
pol_lstm_v02m.py:
    Slightly  Modified version from v02..

    (1) Apply strides in cnn for static-datasets
    (2) Apply second-lstm for time-series..

pol_lstm_v02.py:
    DeepLearning Code for polyol-Process-data modeling for in-silico
    Jae-Min Shin@MCG
    2019-02-26


    **Note>

    <v02>
    Test of LSTM  for Time-Series-Data... : Does not perform well..


"""
import os, sys, glob
import numpy as np, pandas as pd

import keras
from keras.models import Model
from keras.layers import Input, Activation, Reshape, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, LSTM, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import r2_score

Fname_x1 = "../polyol.tsv"
Fname_xt = glob.glob("../data_timeseries/"+"*.csv")

VERSION = "v02m"
path = "./run/%s" % VERSION
if not os.path.exists(path):
    os.makedirs(path)

File_Model = os.path.join(path, "model.bin")
File_Prd = os.path.join(path, "prd.csv")

class Data_XY:
    def __init__(self, tag, y1, y2, x1):
        self.tag = tag[:-8]     ### Up to 'T'
        self.y1 = y1
        self.y2 = y2
        self.Xp = np.array(x1, dtype=np.float)

    def ChkIn_Xt(self, df):
        self.Xt = np.array(df.iloc[:,2:], dtype=np.float)

    def Info(self):
        print "#Data_X1>%s, %.2f, %.3f" % (self.tag, self.y1, self.y2)
        print "\tXp.shape=", self.Xp.shape
        print "\tXt.shape=", self.Xt.shape


def get_dic_xp():
    """
    return static_data, x1


    JMS NOte..>>
      1. There are no-time-series.. data for the last-sample.. So simply ignore last one..
      2. To use time-series, ignore 20-th-coluymn (number of tics??)

      3**. Column-3 is label.., Thus we can use one-hot--encoding 
    """

    df = pd.read_csv(Fname_x1, sep='\t', skiprows=1)
    l_tags = df.iloc[:-1,0]
    l_y1 = df.iloc[:-1,1]
    l_y2 = df.iloc[:-1,2]
    #x1 = df.iloc[:-1,4:19]     ### Without Time-Series-Mean/Std/...
    x1 = df.iloc[:-1,4:]        ### Use all

    print "Before one-hot, x1.shape=", x1.shape

    ### Seosor-label at Column-3
    xl = df.iloc[:-1,3]
    xl_tags = list(set(xl))
    dxl = dict((c,i) for i,c in enumerate(xl_tags))
    x0 = []
    for x in xl:
        x0.append(dxl[x])
    x0h = pd.get_dummies(x0).values
    x1 = np.hstack((x1, x0h))

    print "x1.shape=", x1.shape

    d_x1 = {}
    l_x1 = []
    for i, tag in enumerate(l_tags):
        dx1 = Data_XY(tag, l_y1[i], l_y2[i], x1[i])
        d_x1[dx1.tag] = dx1
        l_x1.append(dx1)

    return d_x1, l_x1


### Check-In Time-Series Data..
def chkin_xt(dic_xy):
    for fn in Fname_xt:
        d,f = os.path.split(fn)
        tag = f[:-10]
        if not tag in dic_xy.keys():
            continue
        df = pd.read_csv(fn, skiprows=1)
        dic_xy[tag].ChkIn_Xt(df)


def get_dxy():
    """
    return l_y, l_x1, l_xt:
    l_y: Target
    l_x1: Static-Data
    l_xt: Time-Series
    """
    ## (1) get static-data
    d_xp, l_xp = get_dic_xp()


    ## (2) ChkIn Time-series..
    chkin_xt(d_xp)

    """
    ############ Confirmed.., Total 185-samples.. #############
    for i,x in enumerate(l_xp):
        x.Info()
    """

    ### Retuen Y, X1, Xt
    l_y, l_x1, l_xt = [], [], []
    for i, xy in enumerate(l_xp):
        l_y.append(xy.y1)   ### First test for OH??
        l_x1.append(xy.Xp)
        l_xt.append(xy.Xt)
    l_xt = pad_sequences(l_xt, dtype='float')

    """
    JMS>Confirmed...
    print "Chk.. pad_sequences..>>>"
    print "l_xt[0]=", l_xt[0]
    print "l_xt[-1]=", l_xt[-1]
    """

    return l_y, l_x1, l_xt
    


### Custom-Callback-Function..
class MyCallBack(keras.callbacks.Callback):
    def __init__(self, model, l_y, l_x1, l_xt):
        self.model = model
        self.y = l_y
        self.x1 = l_x1
        self.xt = l_xt

    def save_prd(self, epoch, y, fx):
        fp = open(File_Prd, 'w')
        fp.write("#Prd>%d\n" % epoch)
        for i in range(len(y)):
            fp.write("%.3f,%.3f\n" % (y[i], fx[i]))
        fp.close()

    def on_epoch_end(self, epoch, logs={}):
        fx = self.model.predict([self.x1, self.xt])
        p_r2 = r2_score(self.y, fx)
        print "###JMS>>>MCB>%d, r2=%.5f, p_loss=%.3f, t_loss=%.3f" % (epoch, p_r2, logs['val_loss'], logs['loss'])
        self.model.save(File_Model)
        self.save_prd(epoch, self.y, fx)
        

def init_model(NX):
    """
    Initialize keras-DL model
    """

    ### (1)  X1..    ===> Static-X
    inX1 = Input(shape=(NX,), name="inX1")       ## Static data-Use all time-series-mean/std..

    x1 = BatchNormalization(name='x1_norm')(inX1)
    x1 = Reshape((-1,1), name="x1_reshape_1")(x1)
    x1 = Conv1D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu', name='x1_conv_1')(x1)
    x1 = Dropout(0.3, name='x1_drop_1')(x1)

    x1 = Conv1D(filters=8, kernel_size=3, strides=2, padding='valid', activation='relu', name='x1_conv_2')(x1)
    x1 = Dropout(0.3, name='x1_drop_2')(x1)

    x1 = Conv1D(filters=8, kernel_size=3, strides=2, padding='valid', activation='relu', name='x1_conv_3')(x1)
    x1 = Dropout(0.3, name='x1_drop_3')(x1)

    x1 = Flatten(name='x1_flat_1')(x1)

    ### (2)  Xt..    ===> Time-Series-RNN
    inXt = Input(shape=(None, 6), name='inXt')      ## (NOne, 6) means (time-steps, 6-tags)
    xt = BatchNormalization(name='xt_norm')(inXt)
    xt = LSTM(8, return_sequences=True, stateful=False, dropout=0.3, recurrent_dropout=0.3, name='xt_lstm-1')(xt)    ## return shape = (?, 16)
    xt = LSTM(8, return_sequences=False, stateful=False, dropout=0.3, recurrent_dropout=0.3, name='xt_lstm-2')(xt)    ## return shape = (?, 16)
    xt = Reshape((-1,1), name='xt-reshape')(xt)

    xt = Conv1D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu', name='xt_conv_1')(xt)
    xt = Dropout(0.3, name='xt_cnn_drop_1')(xt)

    xt = Flatten(name='xt_flat_1')(xt)


    ### Merge (1) and (2)
    x = keras.layers.concatenate([x1, xt])   
    
    x = Dense(8, kernel_initializer='normal', activation='relu', name='merged_dense_1')(x)
    x = Dropout(0.3, name='x_drop_1')(x)
    
    x = Dense(4, kernel_initializer='normal', activation='relu', name='merged_dense_2')(x)

    fx = Dense(1, activation='linear', bias_initializer=keras.initializers.Constant(28.6), name='fx')(x)
    model = Model(inputs=[inX1, inXt], output=fx)
    model.compile(loss='mse',
            optimizer='adam',
            metrics=['mae', 'mse'])
    print model.summary()
    return model



def doit():
    ### Get data
    l_y, l_x1, l_xt = get_dxy()

    ### Input vector dim is 58...
    model = init_model(58)

    ### Define Call-back
    mcb = MyCallBack(model, l_y, l_x1, l_xt)
    e_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
            min_delta=0, patience=500,
            verbose=0, mode='auto')

    model.fit([l_x1, l_xt], l_y, 
            validation_split=0.2, 
            batch_size=128, epochs=12000, 
            verbose=1,
            callbacks=[mcb, e_stop])

    model.save(File_Model)
    print model.summary()


if __name__ == '__main__':
    doit()
