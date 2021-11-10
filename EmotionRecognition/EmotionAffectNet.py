import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from NetworkAffectNet import Network
import tensorflow as tf
import os
RECORD_RIR='../DataSet/AffectNet/AffectNetRecords_64x64_gray_5/'
RECORDS_SPLITS=['Train','Validation','Test']
RECORDS_PER_SPLIT={'Train':5000,'Validation':1000,'Test':1000}
PATTERN ="*_AffectNet.tfrecords"
SAVE_PATH ='../SavedModels/AffectNet/AffectNet64x64_7/'
initial_epoch=00

LOAD_PATH ='../SavedModels/AffectNet/AffectNet64x64_4/AffNet_'+str(initial_epoch)+'.h5'
LABELS=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']
NUM_CLASSES = len(LABELS)

counts= {split:sum([1 for f in os.listdir(os.path.join(RECORD_RIR,split)) if os.path.isfile(os.path.join(os.path.join(RECORD_RIR,split), f))])  for split in RECORDS_SPLITS}

count_Training_data=int(counts['Train']*RECORDS_PER_SPLIT['Train']/50)
count_Validation_data=int(counts['Train']*RECORDS_PER_SPLIT['Validation']/50)
count_Test_data=int(counts['Train']*RECORDS_PER_SPLIT['Test']/50)


shuffle_buffer =15
batch_size = 128
epochs = 10
width, height = 64, 64
model = Network(
    num_labels=NUM_CLASSES,
    width=width,
    height=height
)
model.create_graph()
#
# #model.load_model(path=LOAD_PATH,run=initial_epoch)
model.train(initial_epoch=initial_epoch,
            epochs=epochs,
            batch_size=batch_size,
            count_train_data=count_Training_data,
            count_val_data=count_Validation_data,
            data_dir=RECORD_RIR,
            file_patterns=PATTERN,
            save_path=SAVE_PATH,
            shuffle_buffer=shuffle_buffer)
#model.save_model(path=SAVE_PATH,initial_epoch=initial_epoch)
#model.evaluate(batch_size=batch_size,count_test=count_Test_data,data_dir=RECORD_RIR,file_patterns=PATTERN)
