import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from tensorflow import keras
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D,Input
from tensorflow.keras.losses import categorical_crossentropy
from data.AffectNetTotfRecord import get_dataset
import glob


RECORD_RIR="dataSets/AffectNetRecords_64x64_gray/"
PATTERN ="*_AffectNet.tfrecords"
class Network():
    def __init__(self,num_labels,width,height):
        self.num_labels = num_labels
        self.width = width
        self.height = height
    def creat_graph(self,):



        model_input = keras.layers.Input(shape=(self.width,self.height,1))

        Conv_Layer_1 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(model_input)
        Conv_Layer_2 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(Conv_Layer_1)
        Pool_Layer_1 = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(Conv_Layer_2)

        Conv_Layer_3 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(Pool_Layer_1)
        Drop_Layer_1 = keras.layers.Dropout(0.4)(Conv_Layer_3)
        Conv_Layer_4 = keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(Drop_Layer_1)

        Batch_Norm_1 = keras.layers.BatchNormalization()(Conv_Layer_4)
        Conv_Layer_5 = keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(Batch_Norm_1)
        Batch_Norm_2 = keras.layers.BatchNormalization()(Conv_Layer_5)

        Conv_Layer_6 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(Batch_Norm_2)
        Drop_Layer_2 = keras.layers.Dropout(0.4)(Conv_Layer_6)
        Conv_Layer_7 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(Drop_Layer_2)
        Pool_Layer_2 = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(Conv_Layer_7)
        Conv_Layer_8 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(Pool_Layer_2)

        Flat_Layer = Flatten()(Conv_Layer_8)

        Dense_layer_1 = keras.layers.Dense(1024, activation='relu')(Flat_Layer)
        Output_Layer = keras.layers.Dense(self.num_labels, activation='softmax')(Dense_layer_1)

        #Create your model
        self.model = keras.models.Model(inputs=model_input, outputs=Output_Layer)
        self.model.summary()

        #Compile your model
        self.model.compile( optimizer=keras.optimizers.RMSprop(lr=0.0003),
                            loss=categorical_crossentropy,
                            metrics=['accuracy'])
    def Train(self,epochs,batch_size,count_data):
        #Training the model
        Train_data,Val_data = get_dataset(batch_size,count_data,tfr_dir=RECORD_RIR,pattern=PATTERN)

        train_iterator = Train_data.make_one_shot_iterator()
        val_iterator = Val_data.make_one_shot_iterator()

        self.train_images, self.train_labels,_,_ = train_iterator.get_next()
        self.val_images, self.val_labels,_,_ = val_iterator.get_next()

        # set the pictures to the the proper dimentions
        self.train_input = tf.reshape(self.train_images, [-1, self.width,self.height, 1])
        self.val_input = tf.reshape(self.val_images, [-1, self.width,self.height, 1])

        # Create a one hot array for the labels
        self.train_labels = tf.one_hot(self.train_labels, self.num_labels)
        self.val_labels = tf.one_hot(self.val_labels, self.num_labels)

        with tf.device("/gpu:0"):
            self.model.fit(x=self.train_input,y=self.train_labels,validation_data=(self.val_input,self.val_labels),epochs=epochs,steps_per_epoch =int( count_data/batch_size))
    # def Train(self,epochs,batch_size):
    #     #Training the model
    #     dataSet = get_dataset(batch_size,tfr_dir=RECORD_RIR,pattern=PATTERN)
    #     iterator = dataSet.make_one_shot_iterator()
    #     self.images, self.labels,_,_ = iterator.get_next()
    #     # set the picture to the the proper dimentions
    #     self.input = tf.reshape(self.images, [-1, self.width,self.height, 1])
    #     # Create a one hot array for your labels
    #     self.labels = tf.one_hot(self.labels, self.num_labels)
    #
    #     with tf.device("/gpu:0"):
    #         self.model.fit(self.input,self.labels,epochs=epochs,steps_per_epoch =int( self.count_data/batch_size))

    def SaveModel(self,path):
        fer_json = self.model.to_json()
        with open(path+"fer.json", "w") as json_file:
            json_file.write(fer_json)
            self.model.save_weights("fer.h5")

    def predict(self,image):
        #Training the model
        with tf.device("/gpu:0"):
            return self.model.predict(image)

