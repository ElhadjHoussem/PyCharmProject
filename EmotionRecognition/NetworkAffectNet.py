import os.path
import warnings


warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Activation, \
    Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Input
from tensorflow.python.keras.losses import categorical_crossentropy

from data.DataSetModule import generate_data
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras import regularizers
import numpy as np
import glob

class Network():
    def __init__(self, num_labels, width, height):
        self.num_labels = num_labels
        self.width = width
        self.height = height
        self.regularization_rate = 1e-5
        self.drop_rate = 0.2
        tf.disable_eager_execution()

    def create_graph(self ):
        """---Input---- 64x64 """
        model_input = Input(shape=(self.width, self.height, 1), name='input')

        conv_layer_1= keras.layers.Conv2D(filters=16,kernel_size=(7, 7),activation='relu',padding='same',name="conv_layer_1" )(model_input)

        restNet_1 = self.resNetBlock(input_net=conv_layer_1, num_filter=16, kernel_size=3, activation='relu', padding='same', name="1")
        restNet_2 = self.resNetBlock(input_net=restNet_1, num_filter=16, kernel_size=3, activation='relu', padding='same', name="2")
        restNet_3 = self.resNetBlock(input_net=restNet_2, num_filter=16, kernel_size=3, activation='relu', padding='same', name="3")
        restNet_4 = self.resNetBlock(input_net=restNet_3, num_filter=16, kernel_size=3, activation='relu', padding='same', name="4")
        restNet_5 = self.resNetBlock(input_net=restNet_4, num_filter=16, kernel_size=3, activation='relu', padding='same', name="5",regulization_rate=1e-4)

        Dropout_1 = keras.layers.Dropout(rate=0.2)(restNet_5)
        conv_layer_2= keras.layers.Conv2D(filters=32,kernel_size=(3, 3),activation='relu',padding='same',name="conv_layer_2" )(Dropout_1)
        Pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_layer_2)

        restNet_6 = self.resNetBlock(input_net=Pooling_1, num_filter=32, kernel_size=3, activation='relu', padding='same',name="6")
        restNet_7 = self.resNetBlock(input_net=restNet_6, num_filter=32, kernel_size=3, activation='relu', padding='same',name="7")
        restNet_8 = self.resNetBlock(input_net=restNet_7, num_filter=32, kernel_size=3, activation='relu', padding='same',name="8")
        restNet_9 = self.resNetBlock(input_net=restNet_8, num_filter=32, kernel_size=3, activation='relu', padding='same',name="9",regulization_rate=1e-4)

        Drop_Layer_3 = keras.layers.Dropout(rate=0.1)(restNet_9)
        conv_layer_3= keras.layers.Conv2D(filters=64,kernel_size=(3, 3),activation='relu',padding='same',name="conv_layer_3" )(Drop_Layer_3)
        Pooling_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_layer_3)

        restNet_10 = self.resNetBlock(input_net=Pooling_3, num_filter=64, kernel_size=3, activation='relu', padding='same',name="10")
        restNet_11 = self.resNetBlock(input_net=restNet_10, num_filter=64, kernel_size=3, activation='relu', padding='same',name="11")
        restNet_12 = self.resNetBlock(input_net=restNet_11, num_filter=64, kernel_size=3, activation='relu', padding='same',name="12")
        restNet_13 = self.resNetBlock(input_net=restNet_12, num_filter=64, kernel_size=3, activation='relu', padding='same',name="13",regulization_rate=1e-4)

        Drop_Layer_4 = keras.layers.Dropout(rate=0.1)(restNet_13)
        Pool_Layer_4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(Drop_Layer_4)


        Flat_Layer = Flatten()(Pool_Layer_4)
        FC_layer = keras.layers.Dense(128, activation='relu',name='FC_layer',activity_regularizer=regularizers.l2(1e-5))(Flat_Layer)

        Output_Layer = keras.layers.Dense(self.num_labels, activation='softmax',name='output')(FC_layer)

        """--- Model Build ---"""
        self.model = keras.models.Model(inputs=model_input, outputs=Output_Layer)
        self.model.summary()
        self.model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                           loss=categorical_crossentropy,
                           metrics=['accuracy'])
        #self.model.save('../SavedModels/AffectNet/AffectNet64x64_5/AffNet_02.h5')
    def resNetBlock(self, input_net, num_filter=64, kernel_size=5, activation='relu', padding='same', name="0",regulization_rate=None):

        regularizer=None if regulization_rate is None else regularizers.l2(regulization_rate)
        Conv_Layer_1 = keras.layers.Conv2D(
            num_filter, kernel_size=(kernel_size, kernel_size), activation=activation,
            padding=padding,activity_regularizer=regularizer, name="ResNet_block_" + name + '_1'
        )(input_net)

        Conv_Layer_2 = keras.layers.Conv2D(
            num_filter, kernel_size=(kernel_size, kernel_size), activation=activation,
            padding=padding,activity_regularizer=regularizer, name="ResNet_block_" + name + '_2'
        )(Conv_Layer_1)

        skip_connection_1_2 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([input_net, Conv_Layer_2])

        return skip_connection_1_2


    def train(self, initial_epoch, epochs, batch_size, count_train_data, count_val_data, data_dir, file_patterns,
              save_path, shuffle_buffer):
        tf.disable_eager_execution()

        train_data, val_data, test_data = generate_data(batch_size, shuffle_buffer,tfr_dir=data_dir, pattern=file_patterns)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=save_path + '/AffNet_{epoch:02d}.h5',
            monitor='loss',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,


        )
        lr_scheduler_callbacks = LearningRateScheduler(self.lr_scheduler, verbose=1)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_path,"logs"),
            histogram_freq=1,
            batch_size=batch_size,
            write_graph=True,
            write_images=True,
            update_freq=1000,
            embeddings_freq=0,
            embeddings_metadata=None
        )
        callback_list = [checkpoint_callback, tensorboard_callback, lr_scheduler_callbacks]

        with tf.device("/gpu:0"):
            self.model.fit(
                x=train_data,
                validation_data=val_data,
                shuffle=True,
                epochs=epochs,
                steps_per_epoch=int(count_train_data / batch_size),
                validation_steps=int(count_val_data / batch_size),
                verbose=1,
                callbacks=callback_list,
                initial_epoch=initial_epoch

            )

    def evaluate(self, batch_size, count_test, data_dir, file_patterns):
        _, _, _, _, test_input, val_input = generate_data(batch_size, tfr_dir=data_dir, pattern=file_patterns)
        with tf.device("/gpu:0"):
            self.model.evaluate(
                x=test_input,
                y=val_input,
                batch_size=batch_size,
                steps=int(count_test / batch_size)
            )

    def lr_scheduler(self, epoch, lr):
        decay_rate = 0.85
        decay_step = 2
        if epoch==0:
            lr=0.001
        if epoch % decay_step == 0 and epoch:
            return lr * pow(decay_rate, np.floor(epoch / decay_step))
        return lr

    def predict(self, image):
        # Training the model
        with tf.device("/gpu:0"):
            return self.model.predict(image)

    def load_model(self, path, run):
        self.model = tf.keras.models.load_model(filepath=path)

    def load_model_from_json(self, path):
        # load model
        model = keras.models.model_from_json(open(path + ".json", "r").read())
        # load weights
        model.load_weights(path + '.h5')
        return model

    def save_model(self, path, initial_epoch):
        self.model.save(path + "/AffectNet{}.h5".format(initial_epoch))

    def save_json(self, path):
        json = self.model.to_json()
        with open(path + ".json", "w") as json_file:
            json_file.write(json)









    def create_graph_(self, ):

        regularization_rate = self.regularization_rate
        """---Input---- 64x64 """
        model_input = Input(shape=(self.width, self.height, 1), name='input')

        """ ##############  ########   Block_1  ##########  ########################## """

        '---ResNet  1'
        Conv_Layer_1 = keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(model_input)
        Conv_Layer_2 = keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(Conv_Layer_1)
        skip_connection_1_2 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([Conv_Layer_1, Conv_Layer_2])
        '---Straight ConvNet 1'
        Conv_Layer_3 = keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                           activation='relu', activity_regularizer=regularizers.l2(regularization_rate)
                                           )(skip_connection_1_2)
        Pool_Layer_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(Conv_Layer_3)
        Conv_Layer_4 = keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                           activation='relu', activity_regularizer=regularizers.l2(regularization_rate)
                                           )(Pool_Layer_1)
        Drop_Layer_1 = keras.layers.Dropout(rate=0.2)(Conv_Layer_4)

        '##############  ########   Block_2  ##########  ##########################'

        '--- ResNet 2 --- '
        Conv_Layer_5 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(Drop_Layer_1)
        Conv_Layer_6 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(Conv_Layer_5)
        skip_connection_5_6 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([Conv_Layer_5, Conv_Layer_6])
        '--- Straight ConvNet 2 ---'
        Conv_Layer_7 = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                           activation='relu', activity_regularizer=regularizers.l2(regularization_rate)
                                           )(skip_connection_5_6)
        Pool_Layer_2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(Conv_Layer_7)
        Conv_Layer_8 = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                           activation='relu', activity_regularizer=regularizers.l2(regularization_rate)
                                           )(Pool_Layer_2)
        Drop_Layer_2 = keras.layers.Dropout(rate=0.2)(Conv_Layer_8)

        '##############  ########   Block_3  ##########  ##########################'

        '--- ResNet 3 --- '
        Conv_Layer_9 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(Drop_Layer_2)
        Conv_Layer_10 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(Conv_Layer_9)
        skip_connection_9_10 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([Conv_Layer_9, Conv_Layer_10])

        '--- Straight ConvNet 3 ---'
        Conv_Layer_11 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            activation='relu', activity_regularizer=regularizers.l2(regularization_rate)
                                            )(skip_connection_9_10)
        Pool_Layer_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(Conv_Layer_11)
        Conv_Layer_12 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            activation='relu', activity_regularizer=regularizers.l2(regularization_rate)
                                            )(Pool_Layer_3)
        Drop_Layer_3 = keras.layers.Dropout(rate=0.1)(Conv_Layer_12)

        '##############  ########   Block_4  ##########  ##########################'

        '--- ResNet 4 --- '
        Conv_Layer_13 = keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(Drop_Layer_3)
        Conv_Layer_14 = keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(Conv_Layer_13)
        skip_connection_13_14 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([Conv_Layer_13, Conv_Layer_14])
        '--- Straight ConvNet 4 ---'
        Conv_Layer_15 = keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            activation='relu', activity_regularizer=regularizers.l2(regularization_rate)
                                            )(skip_connection_13_14)
        Pool_Layer_4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(Conv_Layer_15)
        Conv_Layer_16 = keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            activation='relu', activity_regularizer=regularizers.l2(regularization_rate)
                                            )(Pool_Layer_4)
        Drop_Layer_4 = keras.layers.Dropout(rate=0.1)(Conv_Layer_16)

        '##############  ########   Block_5  ##########  ##########################'

        '--- ResNet 5 --- '
        Conv_Layer_17 = keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(Drop_Layer_4)
        Conv_Layer_18 = keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(Conv_Layer_17)
        skip_connection_17_18 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([Conv_Layer_17, Conv_Layer_18])

        '##############  ########   Output block  ##########  ##########################'
        Flat_Layer = Flatten()(skip_connection_17_18)
        Dense_layer_1 = keras.layers.Dense(256, activation='relu',
                                           activity_regularizer=regularizers.l2(regularization_rate))(Flat_Layer)
        Dense_layer_2 = keras.layers.Dense(128, activation='relu',
                                           activity_regularizer=regularizers.l2(regularization_rate))(Dense_layer_1)
        Output_Layer = keras.layers.Dense(self.num_labels, activation='softmax')(Dense_layer_2)

        '---model build---'
        self.model = keras.models.Model(inputs=model_input, outputs=Output_Layer)
        self.model.summary()

        # Compile your model
        self.model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                           loss=categorical_crossentropy,
                           metrics=['accuracy'])
        self.model.save('mnist.h5')
