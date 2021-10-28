import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from NetworkAffectNet import Network
from data.AffectNetTotfRecord import count_annotation


RECORD_RIR="AffectNetRecords_64x64_gray/"
LABELS=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt','None','Uncertain','Non_Face']
NUM_CLASSES = len(LABELS)
count_data=count_annotation()
batch_size = 128
epochs = 50
width, height = 64, 64

model = Network(num_labels=NUM_CLASSES,width=width,height=height)
model.creat_graph()
model.Train(epochs=epochs,batch_size=batch_size,count_data=count_data)

