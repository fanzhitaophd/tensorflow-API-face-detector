"""
Convert xml to csv file

source and credits:
https://raw.githubusercontent.com/datitran/raccoon_dataset/master/xml_to_csv.py
"""

import os
import glob
import pandas as pd 
import xml.etree.ElementTree as ET

#-------------------------------------------#
# Subfunction
#-------------------------------------------#
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot() 
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def train():
    print('Converting training data ...')
    image_path = os.path.join(os.getcwd(), 'Data', 'tf_wider_train',
                              'annotations', 'xmls')
    labels_path = os.path.join(os.getcwd(), 'Data', 'tf_wider_train', 'train.csv')
    if os.path.exists(labels_path) == True:
        print("train.csv already exists!")
        return 
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(labels_path, index=None)
    print('> tf_wider_train - Successfully converted xml to csv.')


def val():
    print('Converting val data ...')
    image_path = os.path.join(os.getcwd(), 'Data', 'tf_wider_val',
                              'annotations', 'xmls')
    labels_path = os.path.join(os.getcwd(), 'Data', 'tf_wider_val', 'val.csv')
    if os.path.exists(labels_path) == True:
        print("val.csv already exists!")
        return 
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(labels_path, index=None)
    print('> tf_wider_val - Successfully converted xml to csv.')
    
#-------------------------------------------#
# Subfunction
#-------------------------------------------#
train() 
val() 
