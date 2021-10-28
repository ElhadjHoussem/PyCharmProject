import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import os

os.environ['TF_KERAS'] = '1'

import keras2onnx
import onnx


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Configures the model for training
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    return model


path = "D:/workspaces/PycharmWS/EmoReha/GestureRecognition/savedModels"
json_config_model = "/handrecognition_model.json"
h5_model = "/handrecognition_model.h5"

reconstructed_model = keras.models.load_model(path + h5_model)

onnx_model_name = '/Handrecognition_model.onnx'

onnx_model = keras2onnx.convert_keras(reconstructed_model)
onnx.save_model(onnx_model, path + onnx_model_name)

#
