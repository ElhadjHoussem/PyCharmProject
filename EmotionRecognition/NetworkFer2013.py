import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from tensorflow import keras
import sys, os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

class Network():
    def __init__(self,num_labels,width,height,input_shape):
        self.num_labels = num_labels

        self.width = width
        self.height = height
        self.input_shape=input_shape


    def creat_graph(self,):
        ##designing the cnn
        #1st convolution layer
        self.model = Sequential()

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(self.input_shape)))
        self.model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
        self.model.add(Dropout(0.5))

        #2nd convolution layer
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
        self.model.add(Dropout(0.5))

        #3rd convolution layer
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

        self.model.add(Flatten())

        #fully connected neural networks
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(self.num_labels, activation='softmax'))

        self.model.summary()
        #
        #Compliling the model
        self.model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(),
                      metrics=['accuracy'])

    def Train(self,X_Train,Y_Train, Test_X, Test_Y, batch_size,epochs):
        #Training the model
        with tf.device("/gpu:0"):
            self.model.fit(X_Train, Y_Train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(Test_X, Test_Y),
                      shuffle=True
                      )

    def SaveModel(self,path):
        fer_json = self.model.to_json()
        with open(path+"fer.json", "w") as json_file:
            json_file.write(fer_json)
            self.model.save_weights("fer.h5")

    def predict(self,image):
        #Training the model
        with tf.device("/gpu:0"):
            return self.model.predict(image)
