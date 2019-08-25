import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, concatenate, Input, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dropout, Flatten
from keras import backend as K
import itertools
from math import sqrt
from sklearn.metrics import mean_squared_error

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

a = np.arange(1, 1001)
y = a*5


lx=[]
for i in range(1000):
    lx.append(i*(a/a))


lx=pd.DataFrame(lx)

lX_train,lX_test,y_train,y_test = train_test_split(lx,y,test_size=0.2,random_state=1)


## 2. LSTM model for time series data
inp2 = Input(shape=(1000, 1), name='input2')
#x2 = LSTM(1, return_sequences=False, stateful=False, dropout=0.3, recurrent_dropout=0.3, name='x2_lstm-1')(inp2)
#x2 = Dense(1, activation='sigmoid')(x2)
x2 = LSTM(1, return_sequences=True, stateful=False, dropout=0.3, recurrent_dropout=0.3, name='x2_lstm_1')(inp2)
x2 = Conv1D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu', name='x2_conv_1')(x2)
x2 = Dropout(0.3, name='x2_cnn_drop_1')(x2)
x2 = Flatten(name='x2_flat_1')(x2)
x2 = Dense(16, activation='linear', name='x2_dense_1')(x2)

# 3. Concatenate CNN layer and LSTM layer
#x = concatenate([x2])
#x = Dense(8, kernel_initializer='normal', activation='relu', name='merged_dense_1')(x)
#x = Dropout(0.3, name='x_drop_1')(x)
#x = Dense(4, kernel_initializer='normal', activation='relu', name='merged_dense_2')(x)
fx = Dense(1, activation='linear', name='fx')(x2)

model = Model(inputs=[inp2], output=fx)
model.compile(loss='mse', optimizer='adam', metrics=[r_squared])
model.summary()

history = model.fit(lX_train.values[:,:,np.newaxis], y_train, batch_size=256, epochs=100, validation_split=0.2, verbose=1)

score = model.evaluate(lX_test.values[:,:,np.newaxis], y_test, verbose=0)
pred = model.predict(lX_test.values[:,:,np.newaxis])

true = list(y_test)
pred = list(itertools.chain(*pred))
corr = np.corrcoef([true, pred])**2
np.corrcoef([np.array(true).astype('int'), np.array(pred).astype('int')])
true = np.array(true).astype('float32')

print("test r-squared: ",round(corr[0][1],4))
print('RMSE: '+str(sqrt(mean_squared_error(true, pred))))
