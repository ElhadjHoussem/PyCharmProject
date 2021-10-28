import tflite2onnx
import tensorflow as tf
import tf2onnx


ressource_path_origine= '../Ressources/models/tflite'
model_file_name_org ='/palm_detection'

ressource_path_dest= '../Ressources/models/onnx'
model_file_name_dest ='/palm_detection'

tflite_path = ressource_path_origine+ model_file_name_org
onnx_path = ressource_path_dest + model_file_name_dest
tflite2onnx.convert(tflite_path+".tflite", onnx_path+",onnx")
