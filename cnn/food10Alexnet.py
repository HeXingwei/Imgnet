#-*- coding=utf-8 -*-
__author__ = "Xingwei He"
from keras.utils import np_utils

import Alexnet
import loadData

nb_classes = 10
#load data
X_train,Y_train,X_test,Y_test= loadData.loadData()

#normalize the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

Alex=Alexnet.Alexnet(X_train,Y_train,X_test,Y_test)
Alex.fit()
