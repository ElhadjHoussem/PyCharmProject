import time
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
from PIL import Image,ImageOps
from tqdm import tqdm
import glob
import os
#from tqdm.notebook import tqdm

########################################################################################################################
'''Initial Global Variables'''
########################################################################################################################

LABELS=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']
NUM_CLASSES = len(LABELS)

ZIP_FILE_NAME = "J:/Emotion/AffectNet.zip"
RECORD_RIR="../DataSet/AffectNet/AffectNetRecords_64x64_gray_5/"
ANNOTATION_SUFFIX_KEYS=['aro','val','exp']
ANNOTATION_TYPES={'aro':'float','val' :'float','exp':'int'}
DATA_DICT_KEYS=['image','expression','arousal','valence']

ANNOTATION_MAP={annotation:key  for annotation in ANNOTATION_SUFFIX_KEYS for key in DATA_DICT_KEYS if annotation in key}

IMAGE_SIZE=64
COLORS=['RGB','GRAY']
NORMALIZE_IMAGE=True
COLOR=COLORS[1]

########################################################################################################################
'''HELPER Functions'''
########################################################################################################################
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value is tensor
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
def loop_list_repeat(list):
    while True:
        for item in list:
            yield item
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

########################################################################################################################
''' TF_RECORD Feature Mapping Function'''
########################################################################################################################

'''while writing Record'''
def parse_data_point(DataPoint):

  #define the dictionary -- the structure -- of a single example
  data = {
      'raw_image': _bytes_feature(serialize_array(DataPoint['image'])),
      'height': _int64_feature(int(DataPoint['image'].shape[0])),
      'width': _int64_feature(int(DataPoint['image'].shape[1])),
      'expression': _int64_feature(int(DataPoint['expression'])),
      'arousal': _float_feature(float(DataPoint['arousal'])),
      'valence': _float_feature(float(DataPoint['valence']))
    }
  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))

  return out
'''while reading Record'''
def parse_tfr_element(element):

   #define the dictionary -- the structure -- of a single example
    data = {
        'raw_image' : tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width':tf.io.FixedLenFeature([], tf.int64),
        'expression':tf.io.FixedLenFeature([], tf.int64),
        'arousal':tf.io.FixedLenFeature([], tf.float32),
        'valence':tf.io.FixedLenFeature([], tf.float32)
    }

    content = tf.io.parse_single_example(element, data)

    #get our 'feature'-- our image -- and reshape it appropriately
    raw_image = content['raw_image']
    height = content['height']
    width = content['width']
    expression = content['expression']
    arousal = content['arousal']
    valence = content['valence']


    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    image = tf.reshape(image, shape=[IMAGE_SIZE,IMAGE_SIZE,1])/255

    return image, expression

########################################################################################################################
'''Functions for Loading Data From Zip File'''
########################################################################################################################
def load_annotation_from_zipFile(file_name=ZIP_FILE_NAME,
                                 annotation_Suffix=ANNOTATION_SUFFIX_KEYS):
    annotations={key:[] for key in annotation_Suffix}
    with ZipFile(file_name,'r') as zip_archive:
        for file in zip_archive.namelist():
            paths = file.split(sep='/')
            if paths[1] == 'annotations':
                annotation_suffix = paths[-1].split('_')[-1].split('.')[0]
                if annotation_suffix in annotation_Suffix:
                    annotations[annotation_suffix].append(np.load(zip_archive.open(file)))
    return annotations


def get_label_indexes(file_name=ZIP_FILE_NAME):
    # retrieve the indexes of each data records in the zipfile
    # with correspondance to its label
    # returns dictionary of labels as keys and lists of indexes as values
    # --->  {'label_0': [indexes ..],'label_1':[indexes],..}

    list_label_indexes={str(key):[] for key in range(len(LABELS))}
    with ZipFile(file_name,'r') as zip_archive:
        for file in zip_archive.namelist():
            paths = file.split(sep='/')
            if paths[1] == 'annotations':
                annotation_suffix = paths[-1].split('_')[-1].split('.')[0]
                annotation_index = int(paths[-1].split('_')[0])
                if annotation_suffix == 'exp':
                    exp = np.load(zip_archive.open(file))
                    list_label_indexes[str(exp)].append(annotation_index)
    return list_label_indexes


def generate_data_point_from_zipFile(label_indexes_lists,file_name=ZIP_FILE_NAME,data_dict_keys=DATA_DICT_KEYS,
                                     annotation_Suffix_keys=ANNOTATION_SUFFIX_KEYS, annotation_map=ANNOTATION_MAP,
                                     annotation_types=ANNOTATION_TYPES
                                     ):
    # function generator to load data from zipfile
    # each iteration yields a list of data point (in dictionary form) that contains one element from each class
    # return -->[
    #             {'image':--, 'expr':1, 'arousal':--,'valence':--},
    #             {'image':--, 'expr':2, 'arousal':--, 'valence':--},
    #             ...
    #           ]
    Data_point_list=[]

    gen_index_list=[loop_list_repeat(label_indexes_lists[key]) for key in label_indexes_lists.keys()]
    annotation_dir_path='train_set/annotations/'
    image_dir_path='train_set/images/'
    with ZipFile(file_name,'r') as zip_archive:
        while True:
            indexes = [next(gen_index_list[i]) for i in range(len(gen_index_list))]
            for index in indexes:
                Data_point={key:None for key in data_dict_keys}
                Annotations = {key:None for key in annotation_Suffix_keys}
                for annotation_suffix in annotation_Suffix_keys:
                    path=annotation_dir_path + str(index) + '_'+annotation_suffix+'.npy'
                    Annotations[annotation_suffix]=np.array(np.load(zip_archive.open(path)),dtype=annotation_types[annotation_suffix])
                image_path = image_dir_path + str(index) + '.jpg'
                image_file = zip_archive.open(image_path)
                image = Image.open(image_file)
                if COLOR=='GRAY':
                    image = ImageOps.grayscale(image)

                Data_point['image'] = np.array(image.resize((IMAGE_SIZE,IMAGE_SIZE)))
                for Annotation_key in annotation_Suffix_keys:
                    Data_point[annotation_map[Annotation_key]]=Annotations[Annotation_key]
                Data_point_list.append(Data_point)
            yield Data_point_list
            Data_point_list=[]


# count all data records in a zip file
def count_data(file_name=ZIP_FILE_NAME):
    with ZipFile(file_name,'r') as zip_archive:
        files = zip_archive.namelist()
    return int(len(files)/5)



########################################################################################################################
''' Read/Write  the TFRecord'''
########################################################################################################################
'''writer function'''

def write_data_in_tfr(split='Train',zip_file_name=ZIP_FILE_NAME,tfrecord_filename:str="_AffectNet", chunk_size:int=10, out_dir:str=RECORD_RIR):

    tf.enable_eager_execution()
    ensure_dir(out_dir)
    ensure_dir(out_dir+'/'+split+'/')
    out_dir = out_dir+'/'+split+'/'

    label_indexes_lists = get_label_indexes(zip_file_name)
    if split=='Train':
        label_indexes_lists_split={
           key: label_indexes_lists[key][:int(0.7*len(label_indexes_lists[key]))]
            for key in label_indexes_lists.keys()
        }

    elif split=='Validation':
        label_indexes_lists_split={
           key: label_indexes_lists[key][int(0.7*len(label_indexes_lists[key])):int(0.85*len(label_indexes_lists[key]))]
            for key in label_indexes_lists.keys()
        }
    elif split=='Test':
        label_indexes_lists_split={
           key: label_indexes_lists[key][int(0.85*len(label_indexes_lists[key])):]
            for key in label_indexes_lists.keys()
        }
    else:
        return Exception("must provide split: train , validation or test")

    max_count_labels=max([len(label_indexes_lists_split[key]) for key in label_indexes_lists_split.keys()])


    total_image = max_count_labels*len(LABELS)
    #determine the number of shards (single TFRecord files) we need:
    splits = (total_image//chunk_size) + 1 #determine how many tfr shards are needed
    if total_image%chunk_size == 0:
        splits-=1
    print(f"\nUsing {splits} shard(s) for {total_image} files, with up to {chunk_size} samples per shard")

    file_count = 0
    rest= total_image
    data_gen = generate_data_point_from_zipFile(label_indexes_lists=label_indexes_lists,file_name=zip_file_name)

    for i in tqdm(range(splits),desc="Global Progress All-Files "+" {} -> {}".format(file_count,total_image)):
        current_shard_name = "{}{}_{}{}.tfrecords".format(out_dir, i+1, splits, tfrecord_filename)
        writer = tf.io.TFRecordWriter(current_shard_name)
        current_shard_count = 0
        chunk_size = chunk_size if rest>chunk_size else rest

        for _ in tqdm(range(chunk_size//len(LABELS)),desc="Local Progress File "+ current_shard_name + " {} ->{} ".format(current_shard_count,chunk_size)):

            current_data_list = next(data_gen)

            for current_data in current_data_list:
                #create the required Example representation
                out = parse_data_point(DataPoint=current_data)

                writer.write(out.SerializeToString())
                current_shard_count+=1
                file_count += 1
                rest = total_image - file_count


        writer.close()


    print(f"\nWrote {file_count} elements to TFRecord")

    return file_count

def write_records():

    print("----- start writing Training Data ----- ")
    write_data_in_tfr(split='Train',chunk_size=5000)
    print("**** finish writing Training Data ****")

    print("----- start writing validation Data -----")
    write_data_in_tfr(split='Validation',chunk_size=1000)
    print("**** finish writing validation Data **** ")

    print("----- start writing test Data -----")
    write_data_in_tfr(split='Test',chunk_size=1000)
    print("**** finish writing validation Data ****")

'''Reader function'''
def get_dataset(split,batch_size=64,shuffle_buffer=None,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    tfr_dir=tfr_dir+split+'/'

    file=tf.data.Dataset.list_files(tfr_dir+pattern)

    dataset = tf.data.TFRecordDataset(file).repeat()

    if split=='Train':
        dataset = dataset.shuffle(batch_size*shuffle_buffer).map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).map(
            augment_data,num_parallel_calls=AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)
    elif split=='Validation':
        dataset = dataset.map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)
    else:
        dataset = dataset.map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)

    return dataset


########################################################################################################################
'''Data Generator for the Training'''
########################################################################################################################
def augment_data(*args):


    args = list(args)
    image=tf.expand_dims(args[0],-1)
    image = tf.cond(tf.random.uniform((), 0, 1) > 0.80,
                    lambda: image, lambda: tf.image.random_brightness(image,max_delta=0.1))
    image = tf.cond(tf.random.uniform((), 0, 1) > 0.80,
                    lambda: image, lambda: tf.image.random_contrast(image,lower=0.1,upper=0.2))
    image = tf.cond(tf.random.uniform((), 0, 1) > 0.75,
                    lambda: image, lambda: tf.image.random_flip_left_right(image))
    image = tf.cond(tf.random.uniform((), 0, 1) > 0.75, lambda: image,
                    lambda: tf.image.random_flip_up_down(image))
    image = tf.cond(tf.random.uniform((), 0, 1) > 0.75, lambda: image,
                    lambda: tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)))

    args[0]=tf.squeeze(image,-1)
    args[0] = tf.reshape(image, shape=[IMAGE_SIZE,IMAGE_SIZE,1])

    return args
def generate_data(batch_size,shuffle_buffer,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_tfr_dir=os.path.join(tfr_dir,'Train')
    val_tfr_dir=os.path.join(tfr_dir,'Validation')
    test_tfr_dir=os.path.join(tfr_dir,'Test')

    train_files=tf.data.Dataset.list_files(train_tfr_dir+pattern)
    val_files=tf.data.Dataset.list_files(val_tfr_dir+pattern)
    test_files=tf.data.Dataset.list_files(test_tfr_dir+pattern)

    train_dataset = tf.data.TFRecordDataset(train_files)
    val_dataset = tf.data.TFRecordDataset(val_files)
    test_dataset = tf.data.TFRecordDataset(test_files)

    ##You should use `dataset.take(k).cache().repeat()` instead.
    train_dataset = train_dataset.shuffle(batch_size*shuffle_buffer).map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).map(
            augment_data,num_parallel_calls=AUTOTUNE).repeat().batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = val_dataset.map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).repeat().batch(batch_size).prefetch(AUTOTUNE)
    test_dataset = test_dataset.map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).repeat().batch(batch_size).prefetch(AUTOTUNE)



    return train_dataset,val_dataset,test_dataset


def generate_data_(batch_size,shuffle_buffer,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_tfr_dir=os.path.join(tfr_dir,'Train')
    val_tfr_dir=os.path.join(tfr_dir,'Validation')
    test_tfr_dir=os.path.join(tfr_dir,'Test')

    train_files=tf.data.Dataset.list_files(train_tfr_dir+pattern)
    val_files=tf.data.Dataset.list_files(val_tfr_dir+pattern)
    test_files=tf.data.Dataset.list_files(test_tfr_dir+pattern)

    train_dataset = tf.data.TFRecordDataset(train_files).repeat()
    val_dataset = tf.data.TFRecordDataset(val_files).repeat()
    test_dataset = tf.data.TFRecordDataset(test_files).repeat()

    train_dataset = train_dataset.shuffle(batch_size*shuffle_buffer).map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).map(
            augment_data,num_parallel_calls=AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = val_dataset.map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)
    test_dataset = test_dataset.map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)



    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()
    test_dataset = test_dataset.make_one_shot_iterator()

    train_images, train_labels,_,_ = train_iterator.get_next()
    val_images, val_labels,_,_ = val_iterator.get_next()
    test_images, test_labels,_,_ = test_dataset.get_next()

    # set the pictures to the the proper dimentions
    train_input = tf.reshape(train_images, [-1, IMAGE_SIZE,IMAGE_SIZE, 1])
    val_input = tf.reshape(val_images, [-1, IMAGE_SIZE,IMAGE_SIZE, 1])
    test_input = tf.reshape(test_images, [-1, IMAGE_SIZE,IMAGE_SIZE, 1])

    # Create a one hot array for the labels
    train_labels = tf.one_hot(train_labels, NUM_CLASSES)
    val_labels = tf.one_hot(val_labels, NUM_CLASSES)
    test_labels = tf.one_hot(test_labels, NUM_CLASSES)

    return train_input,train_labels,val_input,val_labels,test_input,test_labels
########################################################################################################################
'''Testing Tf_Recording '''
########################################################################################################################
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def test_record(split):
    import cv2
    tf.enable_eager_execution()
    file=tf.data.Dataset.list_files(RECORD_RIR+'/'+split+'/'+"*_AffectNet.tfrecords")
    count=0
    for image,exp,valence,aro in tf.data.TFRecordDataset(file).map(parse_tfr_element).take(100):
        print(exp)
        count+=1
        image= cv2.resize(np.array(tensor_to_image(image)),(300,300))
        cv2.putText(image, str(np.array(exp).item()), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        Image._show(Image.fromarray(image))
        time.sleep(2)
    #print("count {}".format(count))
    tf.disable_eager_execution()
    return count
#test_record('Train')
#count_records()

#write_records()

