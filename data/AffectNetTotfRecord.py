import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
from PIL import Image,ImageOps
import tqdm
import glob

#tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
########################################################################################################################
'''Global Variables'''
########################################################################################################################

LABELS=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt','None','Uncertain','Non_Face']
NUM_CLASSES = len(LABELS)

ZIP_FILE_NAME = "J:/Emotion/AffectNet.zip"
RECORD_RIR="AffectNetRecords_64x64_gray/"

ANNOTATION_SUFFIX_KEYS=['aro','val','exp']
ANNOTATION_TYPES={'aro':'float','val' :'float','exp':'int'}
DATA_DICT_KEYS=['image','expression','arousal','valence']

ANNOTATION_MAP={annotation:key  for annotation in ANNOTATION_SUFFIX_KEYS for key in DATA_DICT_KEYS if annotation in key}

IMAGE_SIZE=64
COLORS=['RGB','GRAY']
COLOR=COLORS[1]
########################################################################################################################
'''Functions for Loading Data From Zip File'''
########################################################################################################################

def load_annotation_from_zipFile(file_name,annotation_Suffix):
    annotations={key:[] for key in annotation_Suffix}
    with ZipFile(file_name,'r') as zip_archive:
        for file in zip_archive.namelist():
            paths = file.split(sep='/')
            if paths[1] == 'annotations':
                annotation_suffix = paths[-1].split('_')[-1].split('.')[0]
                if annotation_suffix in annotation_Suffix:
                    annotations[annotation_suffix].append(np.load(zip_archive.open(file)))
    return annotations
def load_data_point_from_zipFile(file_name,data_dict_keys,annotation_Suffix_keys,annotation_map,annotation_types):
    Data_point={key:None for key in data_dict_keys}
    Annotations = {key:None for key in annotation_Suffix_keys}
    with ZipFile(file_name,'r') as zip_archive:
        for file in zip_archive.namelist():
            paths = file.split(sep='/')
            if paths[1] == 'annotations':
                annotation_suffix = paths[-1].split('_')[-1].split('.')[0]
                if annotation_suffix in annotation_Suffix_keys:
                    Annotations[annotation_suffix]=np.load(zip_archive.open(file))
                    Annotation_Loaded = not (None in Annotations.values())
                    if Annotation_Loaded:
                        for annotation_suffix in annotation_Suffix_keys:
                            Annotations[annotation_suffix]=np.array(Annotations[annotation_suffix],dtype=annotation_types[annotation_suffix])
                        image_path = paths[0]+'/images/'+paths[-1].split('_')[0]+'.jpg'
                        image_file = zip_archive.open(image_path)
                        image = Image.open(image_file)
                        if COLOR=='GRAY':
                            image = ImageOps.grayscale(image)
                        Data_point['image'] = np.array(image.resize((IMAGE_SIZE,IMAGE_SIZE)))
                        for Annotation_key in annotation_Suffix_keys:
                            Data_point[annotation_map[Annotation_key]]=Annotations[Annotation_key]

                        yield Data_point
                        Data_point={key:None for key in data_dict_keys}
                        Annotations = {key:None for key in annotation_Suffix_keys}
def load_data_point_from_zipFile_by_chunks(file_name,data_dict_keys,annotation_Suffix_keys,annotation_map,annotation_types,chunck_size=1000):
    Data_Points=[]
    for data_point in load_data_point_from_zipFile(file_name,data_dict_keys,annotation_Suffix_keys,annotation_map,annotation_types):
        Data_Points.append(data_point)
        if len(Data_Points)>=chunck_size:
            yield Data_Points
            Data_Points=[]


def count_annotation(file_name=ZIP_FILE_NAME):
    with ZipFile(file_name,'r') as zip_archive:
        files = zip_archive.namelist()

    return int(len(files)/5)

########################################################################################################################
'''TF_RECORD HELPER Functions'''
########################################################################################################################
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

########################################################################################################################
''' TF_RECORD Feature Mapping Function'''
########################################################################################################################

'''while writing Record'''
def parse_single_image(DataPoint):

  #define the dictionary -- the structure -- of a single example
  data = {
      'raw_image': _bytes_feature(serialize_array(DataPoint['image'])),
      'height': _int64_feature(DataPoint['image'].shape[0]),
      'width': _int64_feature(DataPoint['image'].shape[1]),
      'expression': _int64_feature(int(DataPoint['expression'])),
      'arousal': _float_feature(float(DataPoint['arousal'])),
      'valence': _float_feature(float(DataPoint['valence']))
    }
  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))

  return out

'''while reading Record'''
def parse_tfr_element(element):

    data = {
        'raw_image' : tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width':tf.io.FixedLenFeature([], tf.int64),
        'expression':tf.io.FixedLenFeature([], tf.int64),
        'arousal':tf.io.FixedLenFeature([], tf.float32),
        'valence':tf.io.FixedLenFeature([], tf.float32)
    }

    content = tf.io.parse_single_example(element, data)

    raw_image = content['raw_image']
    height = content['height']
    width = content['width']
    expression = content['expression']
    arousal = content['arousal']
    valence = content['valence']


    #get our 'feature'-- our image -- and reshape it appropriately
    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    image = tf.reshape(image, shape=[height,width])

    return image, expression,arousal,valence

########################################################################################################################
''' Read/Write  the TFRecord'''
########################################################################################################################

'''writer function'''
def write_images_to_tfr(tfrecord_filename:str="_AffectNet", chunk_size:int=10, out_dir:str=RECORD_RIR):

    total_image = count_annotation(ZIP_FILE_NAME,ANNOTATION_SUFFIX_KEYS)
    #determine the number of shards (single TFRecord files) we need:
    splits = (total_image//chunk_size) + 1 #determine how many tfr shards are needed
    if total_image%chunk_size == 0:
        splits-=1
    print(f"\nUsing {splits} shard(s) for {total_image} files, with up to {chunk_size} samples per shard")

    file_count = 0
    data_gen = load_data_point_from_zipFile(ZIP_FILE_NAME,DATA_DICT_KEYS,ANNOTATION_SUFFIX_KEYS,ANNOTATION_MAP,ANNOTATION_TYPES)

    for i in tqdm.tqdm(range(splits),desc="Global Progress All-Files "+" {} -> {}".format(file_count,total_image)):
        current_shard_name = "{}{}_{}{}.tfrecords".format(out_dir, i+1, splits, tfrecord_filename)
        writer = tf.io.TFRecordWriter(current_shard_name)
        current_shard_count = 0

        for _ in tqdm.tqdm(range(chunk_size),desc="Local Progress File "+ current_shard_name + " {} ->{} ".format(current_shard_count,chunk_size)):

            current_data = next(data_gen)

            #current_data = data[index]

            #create the required Example representation
            out = parse_single_image(DataPoint=current_data)

            writer.write(out.SerializeToString())
            current_shard_count+=1
            file_count += 1

        writer.close()
    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count
'''Reader function'''
def get_dataset(batch_size,data_size,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):

    train_size = int(0.7 * data_size)
    val_size = int(0.3 * data_size)
    file=tf.data.Dataset.list_files(tfr_dir+pattern)

    full_dataset = tf.data.TFRecordDataset(file)
    full_dataset = full_dataset.shuffle(data_size)



    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(val_size)

    train_dataset = train_dataset.repeat()
    val_dataset = val_dataset.repeat()

    train_dataset = train_dataset.shuffle(batch_size*50)
    val_dataset = val_dataset.shuffle(batch_size*10)


    train_dataset = train_dataset.map(parse_tfr_element)
    val_dataset = val_dataset.map(parse_tfr_element)

    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset,val_dataset
def get_dataset_(batch_size,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):
    file=tf.data.Dataset.list_files(tfr_dir+pattern)
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.shuffle(batch_size*50)
    dataset = dataset.repeat()
    dataset = dataset.map(parse_tfr_element)
    dataset = dataset.batch(batch_size)
    return dataset

########################################################################################################################
'''Testing Tf_Recording '''
########################################################################################################################

# #write_images_to_tfr(chunk_size=5000)
# image,expression = get_dataset(128)
# for data in get_dataset(128):
#   print(data.shape)
