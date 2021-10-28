import pandas as pd
import cv2
import random
import numpy as np
import os
import csv
import logging
import threading
import time
from dask import dataframe as dd

print(os.listdir("../Ressources"))


def DataSetToCsv(path,output_path):

    dataframe = pd.DataFrame(columns=['emotion','pixels','usage'])
    dataframe.to_csv(output_path+"/ferAug.csv", index=False, mode='a',header=True)

    emotions = []
    pixels = []
    usages=[]
    Usage_list=['Training','PublicTest']
    dir_name_sub_root=""
    count = 0

    test_or_train=["test","train"]
    for root, dirs, files in os.walk(path):


        for i,t in enumerate(test_or_train):
            if root.find(t)!= -1 :
                usage=Usage_list[i]

        for file in files:
            raw_image = np.array(cv2.imread(os.path.join(root, file))).flatten()
            count+=1

            emotions.append(str(root[-1]))
            pixels.append(" ".join([str(pixel) for pixel in raw_image]))
            usages.append(usage)
            if count>1000:
                count= 0
                dictP_n = [(e,p,u) for e,p,u in zip(emotions,pixels, usages)]

                dataframe = pd.DataFrame.from_records(dictP_n,columns=['emotion','pixels','usage'])

                dataframe.to_csv(output_path+"/ferAug.csv", index=False, mode='a',header=False)
                emotions,pixels, usages =[],[],[]
                print('1000 extracted from' + root )
def MergeCsv(path_list,path_csv):
    sep = ";"
    count = 0
    temp = pd.DataFrame(columns=['emotion','pixels','usage'])
    temp.to_csv(path_csv)
    with open(path_csv, "a+") as targetfile:
        for path in path_list :
            with open(path, "r") as f:
                next(f) # << only if the first line contains headers
                for line in f:
                    targetfile.write(line)
                    count+=1
                    if count%1000==0:
                        print("======Merging %s lines =======", str(count))
def MergeCsv2(path_list,path_csv):
    sep = ";"
    count = 0
    threads = list()
    lock = threading.Lock()
    temp = pd.DataFrame(columns=['emotion','pixels','usage'])
    temp.to_csv(path_csv,index=False,header=True)
    for path in path_list :
        count+=1
        x = threading.Thread(target=MergeThread_function, args=(count,path,path_csv,lock))
        threads.append(x)
        x.start()
    for index, thread in enumerate(threads):
        print("Main    : before joining thread {}.".format( index))
        thread.join()
        print("Main    : thread {} done".format( index))
def shuffleCsv(path,fileName):
    start = time.time()
    dask_df = dd.read_csv(path+fileName)
    dask_df = dask_df.shuffle(dask_df.columns[0])

    dask_df.to_csv(path+"shuffeled_"+fileName)
    end = time.time()
    print("Read csv with dask: ",(end-start),"sec")
def MergeThread_function(index,path,path_to_csv,lock):
    print("Thread {}: starting".format(str(index))  )
    ThreadedMergeCsv(index,path,path_to_csv,lock)
    print("Thread {}: finishing".format(str(index))  )
    print("gloabal merge {} so far".format(str(glb_merge))  )
def ThreadedMergeCsv_(index,path,path_csv,lock):
    count = 0
    with open(path, "r") as f:
        next(f) # << only if the first line contains headers
        for line in f:
            with open(path_csv, "a+") as targetfile:
                targetfile.write(line)
                count+=1

                if count>500:
                    global glb_merge
                    glb_merge+=count
                    print("\nThread{} ======Merging {}: lines ======= from {} ===== glb_merge => {}:".format(str(index), str(count),path,glb_merge))
                    count = 0
def ThreadedMergeCsv(index,path,path_csv,lock):
    count = 0
    for line in pd.read_csv(path,chunksize=1):
        with open(path, 'a') as f:

            dataframe=pd.DataFrame(line)
            # dataframe.to_csv(path_csv,mode="a",header=f.tell()==0)
            dataframe.to_csv(path_csv, index=False, mode='a',header=False)

            count+=1

            if count>500:
                global glb_merge
                glb_merge+=count
                print("\nThread{} ======Merging {}: lines ======= from {} ===== glb_merge => {}:".format(str(index), str(count),path,glb_merge))
                count = 0
def ToCsvThread_function(name,index,path,path_to_csv,chunk_size):
    logging.info("Thread %s: starting", name + str(index))
    ThreadedDataSetToCsv(name,index,path,path_to_csv,chunk_size)
    logging.info("Thread %s: finishing", name + str(index))
    logging.info("glb_count count %s: so far", glb_count)
def ThreadedDataSetToCsv(name,index,path,path_to_csv,chunk_size):
    emotions = []
    pixels = []
    usages=[]
    for root,_, files in os.walk(path,topdown=True):
        for file_counter, file in enumerate(files):
            image = cv2.imread(os.path.join(root, file),cv2.IMREAD_GRAYSCALE)
            #image = cv2.resize(raw_image,48)

            emotions.append(str(index))
            pixels.append(" ".join([str(pixel) for pixel in np.array(image).flatten()]))
            usages.append("Training")
            count = len(emotions)
            if(count>=chunk_size or file_counter == len(files)-1):
                dictP_n = [(e,p,u) for e,p,u in zip(emotions,pixels, usages)]
                dataframe = pd.DataFrame(dictP_n,columns=['emotion','pixels','usage'])
                dataframe.to_csv(path_to_csv, index=False, mode='a',header=False)
                del dataframe
                emotions,pixels, usages =[],[],[]
                global glb_count
                glb_count += count
                logging.info("Thread %s: finished %s batch in %s file ==== glb_count is %s  ", name + str(index),count,path_to_csv, glb_count)
def file_to_lists_of_strings(ressource_path,data_file_name,chunck_size=1000):
    with open(ressource_path+data_file_name, "r") as f:
        raws=[]
        line_index=0
        for line in f:
            if line_index==0:
                line_index+=1
            else:
                col=line[:-1].split(',')
                raws.append([str(col[i]) for i in len(col)])
                if any([col[i]=='' or col[i]=='NaN' for i in range(len(col))]):
                    print('problem'+str(line_index))
                line_index+=1

        return raws
def from_image_directory_to_csv(src_path,path_csv,chunk_size):
    dataframes = [pd.DataFrame(columns=['emotion','pixels','usage']) for i in range(7)]
    print(len(dataframes))
    csv_file_names = []
    for index in range(7):
        csv_file_names.append("/FER2013_Aug_train_"+str(index)+".csv")
        dataframes[index].to_csv(path_csv+csv_file_names[index], index=False, mode='a',header='column_names')

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    threads = list()
    for index in range(7):
        logging.info("Main    : create and start thread %d.", index)
        img_src_path_train = src_path + "/train/"+str(index)
        x = threading.Thread(target=ToCsvThread_function, args=("training_data_",index,img_src_path_train,path_csv+csv_file_names[index],chunk_size))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)

if __name__ == "__main__":
    src_path = "../Ressources/data/Fer2013Augmented"
    glb_count = 0
    glb_merge= 0
    path_csv = "../Ressources/data/Fer2013/Fer_2013_Augmented"
    #shuffleCsv(path_csv+"/","Fer2013Aug.csv")
    #MergeCsv2(path_list=[[path_csv+"/"+file for file in files] for _,_,files in os.walk(path_csv)][0],path_csv=path_csv+"/Fer2013Aug.csv")
    from_image_directory_to_csv(src_path,path_csv,500)

