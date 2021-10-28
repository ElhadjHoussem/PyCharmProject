import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from tensorflow import keras
import sys, os
import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical

from NetworkFer2013 import Network
ressource_path='../Ressources/data/Fer2013/FerAugmented'
data_file_name ='/Fer2013Aug.csv'

X_train,train_y,X_test,test_y=[],[],[],[]


ressource_path='../Ressources/data'
data_file_name ='/fer2013.csv'

df=pd.read_csv(ressource_path+data_file_name)


X_train,train_y,X_test,test_y=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")
num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48




X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=to_categorical(train_y, num_classes=num_labels)
test_y=to_categorical(test_y, num_classes=num_labels)

#cannot produce
#normalizing dataSets between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)


model = NetworkFer2013(num_labels=num_labels,width=width,height=height,input_shape=X_train.shape[1:])
model.creat_graph()
model.Train(X_Train=X_train,Y_Train=train_y,Test_X=X_test,Test_Y=test_y,batch_size=batch_size,epochs=epochs)
