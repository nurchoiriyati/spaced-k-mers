#data 10 genus 111 1111 10001
#Agrobacterium, Bartonella,Bradyrhizobium,Brucella,Methylobacterium,Methylocystis, Ochrobactrum, Rhizobium, Rhodopseudomonas,  sinorhizobium
#classifier : Naive bayes, DNN, CNN

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense, Dropout, Flatten, LSTM
from keras.optimizers import SGD, Adam
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.layers import Conv1D,MaxPooling1D, Reshape
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold

# create network DNN
def create_model(neurons = 5):
    adam = Adam(lr=0.001)
    np.random.seed(123)
    model = Sequential()
    model.add(Dense(neurons, input_shape=(input,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    # Compile the model with Cross Entropy Loss
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
def create_modelCNN(filters = 5, kernel_size = 5):
    np.random.seed(123)
    model = Sequential()
    model.add(Conv1D(filters= 3, kernel_size= 3, padding='valid', input_shape=(1,336), activation='relu', data_format='channels_first')) #input_dim
    model.add(MaxPooling1D(pool_size =2))
    model.add(Conv1D(filters = filters, kernel_size = kernel_size, padding='valid', data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=2,padding='valid'))
    model.add(Flatten())
    ##MLP
    model.add(Dense(225))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    # model.add(Dense(225, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(225, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(11))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def create_modelLSTM():
    np.random.seed(0)
    model = Sequential()
    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.0001)
    model.add(LSTM(50, return_sequences=True, input_shape=(1, 336),activation='relu'))
    #model.add(Dropout(0.5))
    model.add(LSTM(30, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(LSTM(250, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))
    # Compile the model with Cross Entropy Loss
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


input = 336
n_classes = 10
sf_train = pd.read_csv('10000datadengan10kelas_arinimers.csv', delimiter='|')
x = sf_train.drop('class', axis=1).values
y = sf_train['class'].values
y = to_categorical(y)
# x = x.reshape(x.shape[0], 1, fitur)

x, val_x, y, val_y = train_test_split(x, y, test_size = 0.025)
model = KerasClassifier(build_fn=create_modelCNN, epochs=200, batch_size=5, verbose=2)
#Optimizer / Update Rule
#kernel_size  = size of sliding window
#filter = how many different window to use
# kernel_size = [6,7,8,9,10]
# filters = [6,7,8,9,10]
neurons = [5,6,7,8,9,10,15,20,30,40,50,75,100,125,150,175,200,225,250,300]
epochs = [100,150]
# param_grid = dict(filters = filters, kernel_size = kernel_size)
# param_grid = dict(neurons = neurons)
param_grid = dict(epochs = epochs)
kfold = KFold(n_splits = 3, shuffle = True)
grid = GridSearchCV(estimator = model, param_grid = param_grid, refit = True, verbose = 2, cv=kfold, n_jobs=-1)
# grid_result = grid.fit(x, y)
grid_result = grid.fit(x, y, validation_data = (val_x,val_y))
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))