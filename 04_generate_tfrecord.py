"""
Generate tfrecord data

Source without adjustments: 
https://raw.githubusercontent.com/datitran/raccoon_dataset/master/generate_tfrecord.py
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append("/home/zhitao/Documents/Python/tensorflow_models/research")

import os
import io 
import pandas as pd
import tensorflow as tf 

from PIL import Image
from collections import namedtuple
from object_detection.utils import dataset_util 

#------------------------------------------#
# Subfunctions 
#------------------------------------------#
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x))
            for filename, x in zip(gb.groups.keys(), gb.groups)]

def class_text_to_int(row_label):
    if row_label == 'faces':
        return 1
    else:
        None 

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        # print(row)
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    # print(classes)
        
    tf_example = tf.train.Example(features = tf.train.Features(feature = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    return tf_example 
        
def write_data(output_path, images_path, csv_input):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(os.getcwd(), images_path)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    # print(grouped)
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

def train():
    print("Writing training tfrecord ...")
    output_path = "Data/train.record"
    images_path = "Data/tf_wider_train/images"
    csv_input = "Data/tf_wider_train/train.csv" 

    if os.path.exists(output_path):
        print("train.record exists!")
        return 
    write_data(output_path, images_path, csv_input)
    print("train.record is written!")
    
def val():
    print("Writing val tfrecord ...")
    output_path = "Data/val.record"
    images_path = "Data/tf_wider_val/images"
    csv_input = "Data/tf_wider_val/val.csv" 

    if os.path.exists(output_path):
        print("val.record exists!")
        return 
    write_data(output_path, images_path, csv_input)
    print("val.record is written!")

#------------------------------------------#
# Main
#------------------------------------------#
train()
val() 
