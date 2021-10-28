import onnx
import os
import keras2onnx
import tensorflow as tf
from tensorflow.keras.models import model_from_json

os.environ['TF_KERAS'] = '1'

path = "../Ressources/models/keras"
model_name = "/fer"
model = model_from_json(open(path+model_name+".json", "r").read())
# load weights
model.load_weights(path+model_name+".h5")
# model = load_model('fer.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, model_name+".onnx")
