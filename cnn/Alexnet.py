from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D


class Alexnet():

    def __init__(self,X_train,Y_train,X_test,Y_test):
        self.X_train=X_train
        self.Y_train=Y_train
        self.X_test=X_test
        self.Y_test=Y_test
        model = Sequential()
        #conv1
        model.add(Convolution2D(96, 11, 11,
                                border_mode='valid',subsample=(4,4),activation='relu',
                                input_shape=(3, 227, 227)))
        #print model.output_shape
        model.add( MaxPooling2D((3, 3), strides=(2,2)) )
        #print model.output_shape
        #conv2
        model.add( ZeroPadding2D((2,2)))
        model.add(Convolution2D(256,5,5, activation='relu'))

        model.add(MaxPooling2D((3, 3), strides=(2,2)) )
        #conv3
        model.add( ZeroPadding2D((1,1)))
        model.add(Convolution2D(384,3,3, activation='relu'))

        #conv4
        model.add( ZeroPadding2D((1,1)))
        model.add(Convolution2D(384,3,3, activation='relu'))

        #conv5
        model.add( ZeroPadding2D((1,1)))
        model.add(Convolution2D(256,3,3, activation='relu'	))

        model.add(MaxPooling2D((3, 3), strides=(2,2)) )

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.add(Dropout(0.5))
        #model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        self.model=model
        self.model.summary()
    def fit(self):
        self.model.fit(self.X_train[:10], self.Y_train[:10], batch_size=100, nb_epoch=10,show_accuracy=True, verbose=1, validation_data=(self.X_test[:10], self.Y_test[:10]))
        score = self.model.evaluate(self.X_test[:10], self.Y_test[:10], show_accuracy=True, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        self.model.save_weights('../modelWeight/Alexnet_model_weights.h5')