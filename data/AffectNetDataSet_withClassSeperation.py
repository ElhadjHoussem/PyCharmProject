import time
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
from PIL import Image,ImageOps
from tqdm.auto import tqdm
import glob
import os

import tensorflow.contrib.eager as tfe
########################################################################################################################
'''Global Variables'''
########################################################################################################################

LABELS=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']
NUM_CLASSES = len(LABELS)

ZIP_FILE_NAME = "J:/Emotion/AffectNet.zip"
RECORD_RIR="../DataSet/AffectNet/AffectNetRecords_64x64_gray_4/"
ANNOTATION_SUFFIX_KEYS=['aro','val','exp']
ANNOTATION_TYPES={'aro':'float','val' :'float','exp':'int'}
DATA_DICT_KEYS=['image','expression','arousal','valence']

ANNOTATION_MAP={annotation:key  for annotation in ANNOTATION_SUFFIX_KEYS for key in DATA_DICT_KEYS if annotation in key}

IMAGE_SIZE=64
COLORS=['RGB','GRAY']
NORMALIZE_IMAGE=True
COLOR=COLORS[1]
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

def loop_list_repeat(list):
    while True:
        for item in list:
            yield item
def get_label_indexes(file_name=ZIP_FILE_NAME):
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


def count_expression_labels(file_name=ZIP_FILE_NAME):
    annotations = load_annotation_from_zipFile(file_name)
    expressions_count ={}
    for expression in annotations['exp']:
        exp = np.array(expression).item()
        try:
            expressions_count[exp]+=1
        except:
            expressions_count[exp]=1
    return expressions_count


def load_data_point_from_zipFile(file_name=ZIP_FILE_NAME,
                                  data_dict_keys=DATA_DICT_KEYS,
                                  annotation_Suffix_keys=ANNOTATION_SUFFIX_KEYS,
                                  annotation_map=ANNOTATION_MAP,
                                  annotation_types=ANNOTATION_TYPES):
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

def load_data_point_from_zipFile_with_seperate_labels(max_count_labels,label_indexes_lists,
                                                      file_name=ZIP_FILE_NAME,
                                                      data_dict_keys=DATA_DICT_KEYS,
                                                      annotation_Suffix_keys=ANNOTATION_SUFFIX_KEYS,
                                                      annotation_map=ANNOTATION_MAP,
                                                      annotation_types=ANNOTATION_TYPES
                                                      ):
    Data_point_list=[]
    gen_list=[loop_list_repeat(label_indexes_lists[str(i)]) for i in range(len(label_indexes_lists.keys()))]

    annotation_dir_path='train_set/annotations/'
    image_dir_path='train_set/images/'
    with ZipFile(file_name,'r') as zip_archive:
        for _ in range(max_count_labels):
            indexes = [next(gen_list[i]) for i in range(len(gen_list))]
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



def load_data_point_from_zipFile_by_chunks(file_name,data_dict_keys,annotation_Suffix_keys,annotation_map,annotation_types,chunck_size=1000):
    Data_Points=[]
    for data_point in load_data_point_from_zipFile(file_name,data_dict_keys,annotation_Suffix_keys,annotation_map,annotation_types):
        Data_Points.append(data_point)
        if len(Data_Points)>=chunck_size:
            yield Data_Points
            Data_Points=[]

def count_data(file_name=ZIP_FILE_NAME):
    with ZipFile(file_name,'r') as zip_archive:
        files = zip_archive.namelist()

    return int(len(files)/5)


########################################################################################################################
'''TF_RECORD HELPER Functions'''
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
########################################################################################################################
''' TF_RECORD Feature Mapping Function'''
########################################################################################################################
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
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
    image = tf.reshape(image, shape=[height,width])/255

    return image, expression,arousal,valence
########################################################################################################################
''' Read/Write  the TFRecord'''
########################################################################################################################
'''writer function'''
def write_data_in_tfr_from_zip(zip_file_name=ZIP_FILE_NAME,tfrecord_filename:str="_AffectNet", chunk_size:int=10, out_dir:str=RECORD_RIR):
    tf.enable_eager_execution()
    ensure_dir(RECORD_RIR)
    total_image = count_data(zip_file_name)
    #determine the number of shards (single TFRecord files) we need:
    splits = (total_image//chunk_size) + 1 #determine how many tfr shards are needed
    if total_image%chunk_size == 0:
        splits-=1
    print(f"\nUsing {splits} shard(s) for {total_image} files, with up to {chunk_size} samples per shard")

    file_count = 0
    rest= total_image
    data_gen = load_data_point_from_zipFile(zip_file_name,DATA_DICT_KEYS,ANNOTATION_SUFFIX_KEYS,ANNOTATION_MAP,ANNOTATION_TYPES)
    for i in tqdm(range(splits),desc="Global Progress All-Files "+" {} -> {}".format(file_count,total_image)):
        current_shard_name = "{}{}_{}{}.tfrecords".format(out_dir, i+1, splits, tfrecord_filename)
        writer = tf.io.TFRecordWriter(current_shard_name)
        current_shard_count = 0
        chunk_size = chunk_size if rest>chunk_size else rest
        for _ in tqdm(range(chunk_size),desc="Local Progress File "+ current_shard_name + " {} ->{} ".format(current_shard_count,chunk_size)):

            try:
                current_data = next(data_gen)
                exp = int(current_data['expression'])
                print(int(exp))

                #current_data = data[index]

                #create the required Example representation
                out = parse_data_point(DataPoint=current_data)

                writer.write(out.SerializeToString())
                current_shard_count+=1
                file_count += 1
                rest = total_image - file_count
            except:
                print('no more data in zip file')

        writer.close()
    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count

def write_data_in_tfr_from_zip_balanced(zip_file_name=ZIP_FILE_NAME,tfrecord_filename:str="_AffectNet", chunk_size:int=10, out_dir:str=RECORD_RIR):
    tf.enable_eager_execution()
    ensure_dir(RECORD_RIR)
    label_indexes_lists = get_label_indexes(zip_file_name)
    max_count_labels=max([len(label_indexes_lists[key]) for key in label_indexes_lists.keys()])


    total_image = max_count_labels*len(LABELS)
    #determine the number of shards (single TFRecord files) we need:
    splits = (total_image//chunk_size) + 1 #determine how many tfr shards are needed
    if total_image%chunk_size == 0:
        splits-=1
    print(f"\nUsing {splits} shard(s) for {total_image} files, with up to {chunk_size} samples per shard")

    file_count = 0
    rest= total_image
    data_gen = load_data_point_from_zipFile_with_seperate_labels(max_count_labels=max_count_labels,label_indexes_lists=label_indexes_lists,file_name=zip_file_name)

    for i in tqdm(range(splits),desc="Global Progress All-Files "+" {} -> {}".format(file_count,total_image)):
        current_shard_name = "{}{}_{}{}.tfrecords".format(out_dir, i+1, splits, tfrecord_filename)
        writer = tf.io.TFRecordWriter(current_shard_name)
        current_shard_count = 0
        chunk_size = chunk_size if rest>chunk_size else rest
        if i>0:
            break
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
        break

    print(f"\nWrote {file_count} elements to TFRecord")

    return file_count

def write_data_in_tfr_from_zip_seperate(zip_file_name=ZIP_FILE_NAME,tfrecord_filename:str="_AffectNet", chunk_size:int=10, out_dir:str=RECORD_RIR,exp_class=0):
    ensure_dir(RECORD_RIR)
    out_dir=RECORD_RIR+str(exp_class)+'/'
    ensure_dir(out_dir)

    total_image = count_data(zip_file_name)

    file_count = 0
    rest= total_image
    current_record_count=0
    current_class_count=0
    old_file_count=0
    Done=False


    data_gen = load_data_point_from_zipFile(zip_file_name,DATA_DICT_KEYS,ANNOTATION_SUFFIX_KEYS,ANNOTATION_MAP,ANNOTATION_TYPES)
    with tqdm(total=total_image,desc='--Progress Label Class --{}--'.format(LABELS[exp_class]),position=1,leave=True) as global_progress_bar:
        while file_count<total_image-1:
            current_record_count+=1
            current_shard_name = "{}{}{}.tfrecords".format(out_dir, current_record_count, tfrecord_filename)
            writer = tf.io.TFRecordWriter(current_shard_name)
            writer_file_count =0
            chunk_size = chunk_size if rest>=chunk_size else rest
            with tqdm(total=chunk_size,desc="--Progress Shard  --{}--".format(current_record_count),position=2,leave=False) as local_prograss_bar:
                while writer_file_count<chunk_size-1:
                    file_count += 1
                    rest -= 1
                    current_data = next(data_gen,None)
                    if current_data is None:
                        print('writer_file_count {} chunk_size {} file_count {} total_image {} '.
                              format(writer_file_count,chunk_size,file_count,total_image))
                        break;
                    expression_class = int(current_data['expression'])
                    if expression_class == exp_class:
                        out = parse_data_point(DataPoint=current_data)
                        writer.write(out.SerializeToString())
                        writer_file_count+=1
                        current_class_count+=1
                        local_prograss_bar.update(1)
                writer.close()
            global_progress_bar.update(file_count-old_file_count)
            old_file_count=file_count

    print(f"\nWrote {current_class_count} elements of label class {LABELS[exp_class]} to TFRecord")
    return current_class_count

def write_data_in_tfr(chunk_size=5000,mode=None):
    tf.enable_eager_execution()
    with tf.device('cpu'):
        if mode is None:
            write_data_in_tfr_from_zip(chunk_size=chunk_size)
        if mode=='seperate':
            for i in tqdm(range(1,len(LABELS)),desc="--Total Progress writing all Classes --",position=0,leave=True):
                write_data_in_tfr_from_zip_seperate(chunk_size=chunk_size,exp_class=i)
        elif mode=='balanced':
            write_data_in_tfr_from_zip_balanced(chunk_size=chunk_size)


'''Reader function'''
def get_dataset(batch_size,shuffle_buffer,data_size,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_size = int(0.7 * data_size)
    val_size =  int(0.15 * data_size)

    file=tf.data.Dataset.list_files(tfr_dir+pattern)

    full_dataset = tf.data.TFRecordDataset(file)

    train_dataset = full_dataset.take(train_size).cache()
    val_dataset = full_dataset.skip(val_size).cache()
    test_dataset = val_dataset.skip(val_size).cache()

    train_dataset = train_dataset.shuffle(batch_size*shuffle_buffer).map(
        parse_tfr_element,num_parallel_calls=AUTOTUNE).map(
        augment_data,num_parallel_calls=AUTOTUNE).repeat().batch(batch_size).prefetch(AUTOTUNE)


    val_dataset = val_dataset.map(
        parse_tfr_element,num_parallel_calls=AUTOTUNE).repeat().batch(batch_size).prefetch(AUTOTUNE)
    test_dataset = test_dataset.map(
        parse_tfr_element,num_parallel_calls=AUTOTUNE).repeat().batch(batch_size).prefetch(AUTOTUNE)


    return train_dataset,val_dataset,test_dataset
def count_records():
    labels_counts={}
    for label_class in range(len(LABELS)):
        file=tf.data.Dataset.list_files(RECORD_RIR+'/'+str(label_class)+'/'+"*_AffectNet.tfrecords")
        count=0
        for _ in tf.data.TFRecordDataset(file).map(parse_tfr_element).take(-1):
            count+=1
        print("label {} count {}".format(LABELS[label_class],count))
        labels_counts[str(label_class)]=count
    tf.disable_eager_execution()
    return labels_counts

def get_dataset_with_seperated_classe_(batch_size,shuffle_buffer,data_size,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    counts_label_classes=get_count_records()
    train_size_list = [int(0.7 * counts_label_classes[str(l)]) for l in range(len(LABELS))]
    val_size_list =  [int(0.3 * counts_label_classes[str(l)]) for l in range(len(LABELS))]
    files_list = [tf.data.Dataset.list_files(tfr_dir+str(i)+'/'+pattern) for i in range(len(LABELS))]


    full_datasets_list = [tf.data.TFRecordDataset(files) for files in files_list]


    train_dataset_list = [full_datasets_list[l].take(train_size_list[l]) for l in range(len(LABELS))]
    train_dataset_list = [train_dataset_list[l].cache() for l in range(len(LABELS))]


    val_dataset_list = [full_datasets_list[l].skip(train_size_list[l]) for l in range(len(LABELS))]
    val_dataset_list = [val_dataset_list[l].cache() for l in range(len(LABELS))]

    train_dataset_list = [train_dataset_list[l].shuffle(batch_size*shuffle_buffer) for l in range(len(LABELS))]

    train_dataset_list = [train_dataset_list[l].map(parse_tfr_element,num_parallel_calls=AUTOTUNE) for l in range(len(LABELS))]
    train_dataset_list = [train_dataset_list[l].map(augment_data,num_parallel_calls=AUTOTUNE) for l in range(len(LABELS))]

    val_dataset_list = [val_dataset_list[l].map(parse_tfr_element,num_parallel_calls=AUTOTUNE) for l in range(len(LABELS))]

    train_dataset_list = [train_dataset_list[l].batch(batch_size).prefetch(AUTOTUNE).repeat() for l in range(len(LABELS))]
    val_dataset_list = [val_dataset_list[l].batch(batch_size).prefetch(AUTOTUNE).repeat() for l in range(len(LABELS))]

    train_dataset= train_dataset_list[0]
    val_dataset=  val_dataset_list[0]

    for i in range(1,len(train_dataset_list)):
        train_dataset= train_dataset.concatenate(train_dataset_list[i])
    for i in range(1,len(val_dataset_list)):
        val_dataset= val_dataset.concatenate(train_dataset_list[i])

    train_dataset.shuffle(batch_size*shuffle_buffer).batch(batch_size).prefetch(AUTOTUNE)
    val_dataset.batch(batch_size).prefetch(AUTOTUNE)
    return train_dataset,val_dataset

def get_dataset_with_seperated_classe__(batch_size,shuffle_buffer,data_size,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    #counts_label_classes=get_count_records()
    counts_label_classes = {'0': 74874,'1': 134415,'2': 25459,'3': 14090,'4': 6378,'5': 3803,'6': 24882,'7': 3750}

    train_size_list = [int(0.7 * counts_label_classes[str(l)]) for l in range(len(LABELS))]
    files_list = [tf.data.Dataset.list_files(tfr_dir+str(i)+'/'+pattern) for i in range(len(LABELS))]


    full_datasets_list = [tf.data.TFRecordDataset(files).repeat() for files in files_list]


    train_dataset_list = [full_datasets_list[l].take(train_size_list[l]) for l in range(len(LABELS))]

    val_dataset_list = [full_datasets_list[l].skip(train_size_list[l]) for l in range(len(LABELS))]

    pre_batch_sizes_list=[]

    train_pre_batches_list = [
        train_dataset_list[l].map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).batch(int(batch_size/len(LABELS)))
        for l in range(len(LABELS))
    ]
    val_pre_batches_list = [
        val_dataset_list[l].map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE).batch(int(batch_size/len(LABELS)))
        for l in range(len(LABELS))]

    train_dataset= train_pre_batches_list[0]
    val_dataset=  val_pre_batches_list[0]
    for i in range(1,len(LABELS)):
        train_dataset= train_dataset.concatenate(train_pre_batches_list[i])
    for i in range(1,len(LABELS)):
        val_dataset= val_dataset.concatenate(val_pre_batches_list[i])

    train_dataset=train_dataset.shuffle(batch_size).map(
        augment_data,num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    val_dataset=val_dataset.prefetch(AUTOTUNE)

    return train_dataset,val_dataset

def get_dataset_with_seperated_classe(batch_size,shuffle_buffer,data_size,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    #counts_label_classes=get_count_records()
    counts_label_classes = {'0': 74874,'1': 134415,'2': 25459,'3': 14090,'4': 6378,'5': 3803,'6': 24882,'7': 3750}

    train_size_list = [int(0.7 * counts_label_classes[str(l)]) for l in range(len(LABELS))]
    files_list = [tf.data.Dataset.list_files(tfr_dir+str(i)+'/'+pattern) for i in range(len(LABELS))]

    full_datasets_list = [tf.data.TFRecordDataset(files)for files in files_list]

    train_dataset_list = [full_datasets_list[l].take(train_size_list[l]) for l in range(len(LABELS))]
    val_dataset_list = [full_datasets_list[l].skip(train_size_list[l])for l in range(len(LABELS))]
    #batch_splits = random_batches_split(batch_size,len(LABELS))
    train_pre_batches_list = [
        train_dataset_list[l].shuffle(int(batch_size/len(LABELS))*shuffle_buffer).map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE
        ).map(
            augment_data,num_parallel_calls=AUTOTUNE
        ).batch(int(batch_size/len(LABELS))*3).cache().prefetch(AUTOTUNE).repeat()
        for l in range(len(LABELS))
    ]
    val_pre_batches_list = [
        val_dataset_list[l].map(
            parse_tfr_element,num_parallel_calls=AUTOTUNE
        ).batch(int(batch_size/len(LABELS))*3).cache().prefetch(AUTOTUNE).repeat()
        for l in range(len(LABELS))
    ]


    return train_pre_batches_list,val_pre_batches_list
########################################################################################################################
'''Data Generator for the Training'''
########################################################################################################################
def generate_data_(batch_size,shuffle_buffer,data_size,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):
    Train_data_list,Val_data_list = get_dataset_with_seperated_classe(batch_size,shuffle_buffer,data_size,tfr_dir, pattern)


    train_iterators_list = [Train_data_list[i].make_one_shot_iterator() for i in range(len(LABELS))]
    val_iterators_list = [Val_data_list[i].make_one_shot_iterator() for i in range(len(LABELS))]


    training_data = [train_iterators_list[i].get_next() for i in range(len(LABELS))  ]
    val_data= [val_iterators_list[i].get_next() for i in range(len(LABELS))]

    # set the pictures to the the proper dimentions
    train_input = tf.concat(
        [
            tf.reshape(train_mini_batch[0], [-1, IMAGE_SIZE,IMAGE_SIZE, 1]) for train_mini_batch in training_data
        ],0)
    val_input = tf.concat(
        [
            tf.reshape(val_mini_batch[0], [-1, IMAGE_SIZE,IMAGE_SIZE, 1]) for val_mini_batch in val_data
        ],0)

    # Create a one hot array for the labels
    train_labels = tf.concat(
        [
            tf.one_hot(train_mini_batch[1], NUM_CLASSES) for train_mini_batch in training_data
        ],0)
    val_labels =tf.concat(
        [
            tf.one_hot(val_mini_batch[1], NUM_CLASSES) for val_mini_batch in val_data
        ],0)

    return train_input,train_labels,val_input,val_labels

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
    return args
def generate_data(batch_size,shuffle_buffer,data_size,tfr_dir:str=RECORD_RIR, pattern:str="*_AffectNet.tfrecords"):
    Train_data,Val_data,test_dataset= get_dataset(batch_size,shuffle_buffer,data_size,tfr_dir, pattern)


    train_iterator = Train_data.make_one_shot_iterator()
    val_iterator = Val_data.make_one_shot_iterator()
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

    return train_input,train_labels,val_input,val_labels,test_images,test_labels

########################################################################################################################
'''Testing Tf_Recording '''
########################################################################################################################
def test_record_seperate(label_class=0):
    tf.enable_eager_execution()
    file=tf.data.Dataset.list_files(RECORD_RIR+'/'+str(label_class)+'/'+"*_AffectNet.tfrecords")
    count=0
    for image,exp,valence,aro in tf.data.TFRecordDataset(file).map(parse_tfr_element).take(10):
        print(image)
        count+=1
    print("label {} count {}".format(label_class,count))
    tf.disable_eager_execution()
    return
def test_record():
    tf.enable_eager_execution()
    file=tf.data.Dataset.list_files(RECORD_RIR+"*_AffectNet.tfrecords")
    count=0
    for image,exp,valence,aro in tf.data.TFRecordDataset(file).map(parse_tfr_element).take(100):
        print(exp)
        count+=1
    print("count {}".format(count))
    tf.disable_eager_execution()
    return
def random_batches_split(batch_size,num_classes):
    total_batch=batch_size
    min=1
    max = int(batch_size/num_classes)
    batche_sizes_list=[0 for _ in range(num_classes)]
    i=0
    while sum(batche_sizes_list)<batch_size:
        temp_batch_size = np.random.randint(min,max)
        sum_list=sum(batche_sizes_list)
        if temp_batch_size+sum_list>total_batch:
            temp_batch_size = total_batch-sum_list

        batche_sizes_list[i]+=temp_batch_size
        i= (i+1) % num_classes
    np.random.shuffle(batche_sizes_list)
    return np.array(batche_sizes_list)

#test_record()
#write_data_in_tfr(mode='balanced')

#write_data_in_tfr()



