# This program would produce labels for svms
# Input: tfrecord
# Output: 
#         lables n*10 
#         one label would be like [-1,-1,-1,-1,-1,-1,-1,-1,-1,1]

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from os import walk
import json


import tensorflow as tf
#---------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------#
def tftoTxt(dataType):
        filenames = []
        video_lvl_record = []
        for (dirpath, dirnames, filename) in walk("./yt8m_50/tfrecord_dataset/" + dataType + "/"):
                filenames = filename
                

        for file in filenames:
          video_lvl_record.append("./yt8m_50/tfrecord_dataset/"+ dataType +"/" + file)

#---------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------#
        labels = []
        for record in video_lvl_record:
            for example in tf.python_io.tf_record_iterator(record):
                tf_example = tf.train.Example.FromString(example)
                labels.append(tf_example.features.feature['labels'].int64_list.value)

              
#---------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------#
        a = {0,1,3,8,10,12,18,22,23,25} # 10 classes
        #mapclass = {0: "Game", 1: "Vehicle", 3: "Concert", 8: "Association Football", 10: "Animal",
        #                 12:"Food", 18:"Outdoor Recreation", 22: "Nature", 23:"Mobile Phone", 25:"Toy"}
        mapclass = {0: 0, 1: 1, 3: 2, 8: 3, 10: 4, 12:5, 18:6, 22:7, 23:8, 25:9}
        new_labels = []


        for i in range(len(labels)):
            if ( len(set(labels[i]) & a) == 1 ):
                new_labels.append(list(set(labels[i]) & a))

        new_labels = list(map(lambda x: mapclass[x[0]], new_labels))

        svm_labels = [[-1 for i in range(10)] for j in range(len(new_labels))]
        for i in range(len(svm_labels)):
                svm_labels[i][new_labels[i]] = 1

#---------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------#

        with open('./txtdata/yt8m_50_svm'+ dataType +'_labels.txt', 'w') as labelfile:
                json.dump(svm_labels, labelfile)

        labelfile.close()

tftoTxt("train")
tftoTxt("validate")
tftoTxt("test")

