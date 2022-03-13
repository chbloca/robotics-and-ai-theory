# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:20:49 2019

@author: mikys
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def reshape_to_2D(x):
    return x.reshape(x.shape[0],x.shape[1],1)

maxlen, maxcol = train_data.shape
X_train = train_data.reshape(maxlen, maxcol,1)

X_test = test_data.reshape(maxlen_test, data_dim_test,1)

    
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


 
data_dim = maxcol #20
timesteps = maxlen #1420
num_classes = 10
batch_size = 1420

   

model = Sequential()
#    model.add(LSTM(32, return_sequences=True, stateful=True,
#               batch_input_shape=(timesteps, data_dim, batch_size)))
    
   
model.add(LSTM(32, return_sequences=True, stateful=True,
           batch_input_shape=(timesteps, data_dim, batch_size)))
    
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

print(model.summary())



model.fit(X_train, y_train,
      batch_size=batch_size, epochs=1, shuffle=False,
      validation_data=(X_test, y_test))

model.compile(loss='binary_crossentropy',
          optimizer='sgd',
          class_mode="categorical")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=3,
      validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test,
                        batch_size=batch_size,
                        show_accuracy=True)
print('Test score:', score)

print('Test accuracy:', acc)

result_LSTM = model.predict(X_test, verbose=0)
    