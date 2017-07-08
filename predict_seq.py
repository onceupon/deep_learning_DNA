import numpy as np
import os
import tensorflow as tf
import sys
#========================= Model ==============================================
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Convolution3D
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

##Read data
f = open('humanvsran_label', "r")
lines = f.readlines()
f.close()
dat_y = np.zeros((len(lines),2))
for i in range(len(lines)):
	#dat_y[i] = float(dat_y[i].rstrip())
	dat_y[i,int(lines[i].rstrip())] = 1

f = open(sys.argv[1], "r")
lines = f.readlines()
f.close()

import re
def DNA_matrix(seq):
	tem2 = ['[aA]','[cC]','[gG]','[tT]']
	for i in range(len(tem2)):
		ind = [m.start() for m in re.finditer(tem2[i], seq)]
		tem = np.zeros(len(seq),dtype=np.int)
		tem[ind] = 1
		if i==0:
			a = np.zeros((len(seq),4))
		a[...,i] = tem
	return a

for i in range(len(lines)):
	tem = lines[i].rstrip()
	if i==0:
		dat_x = np.zeros((len(lines),len(tem),4))
	dat_x[i,] = DNA_matrix(tem)

import random
ind = range(20000)
random.shuffle(ind)
x_train=dat_x[ind[0:14000]]
#x_train.shape #(1999,251,4)
y_train=dat_y[ind[0:14000]]
x_val=dat_x[ind[14001:20001]]
y_val=dat_y[ind[14001:20001]]
#x_test=dat_x[20001:40001]
#y_test=dat_y[20001:40001]
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
#x_test = x_test.astype('float32')

model=Sequential()
model.add(Conv1D(filters=300,kernel_size=19,strides=1,padding='valid',input_shape=(250,4)))
#print model.output_shape
#model.add(Conv1D(filters=20,kernel_size=10,strides=1,padding='valid'))
model.add(MaxPooling1D(pool_size=10, strides=5, padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=2, mode='min')
history = model.fit(x_train, y_train, batch_size=16, nb_epoch=5, verbose=1, validation_data= None,callbacks=[early_stopping])
print(history.history)
#score = model.evaluate(x_test, y_test, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

weights = model.layers[0].get_weights()
wt=(np.transpose(weights[0][:,:,0]))
wtm=np.transpose(weights[0][:,:,0]).min(axis=0)
wtp=wt-wtm
wtps=np.sum(wtp, axis=0)
print np.round((wtp/wtps)*100)

y_score = model.predict(x_val)
loss, acc = model.evaluate(x_val, y_val, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)

generate_results(y_val[:, 0], y_score[:, 0])
