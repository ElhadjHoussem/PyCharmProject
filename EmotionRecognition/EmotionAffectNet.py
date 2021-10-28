import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from Network import Network
from dataSets.AffectNet import count_annotation,parse_tfr_element,get_dataset
import tensorflow as tf
import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()


RECORD_RIR="AffectNetRecords_64x64_gray/"
LABELS=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt','None','Uncertain','Non_Face']
NUM_CLASSES = len(LABELS)
count_data=count_annotation()
batch_size = 128
epochs = 50
width, height = 64, 64

#count_data= count_annotation()


model = Network(num_labels=NUM_CLASSES,width=width,height=height)
model.creat_graph()

# dataset = get_dataset(batch_size)
# dataset = dataset.batch(5)
# for batch in tfe.Iterator(dataset):
#      print(batch)

model.Train(epochs=epochs,batch_size=batch_size,count_data=count_data)

