"""
Convert the training data and validation data to piscal xml format 
"""

import os
import cv2
import numpy as np
import xml.etree.cElementTree as ET

from glob import iglob
from shutil import copyfile 

#-------------------------------------------------#
# Subfunctions
#-------------------------------------------------#
def newXMLPASCALfile(imgH, imgW, path, basename):
    annotation = ET.Element("annotation", verified = "yes")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = basename
    ET.SubElement(annotation, "path").text = path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "test"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(imgW)
    ET.SubElement(size, "height").text = str(imgH)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    tree = ET.ElementTree(annotation)
    return tree

def appendXMLPASCAL(curr_et_obj, x, y, w, h, filename):
    et_obj = ET.SubElement(curr_et_obj.getroot(), "object")
    ET.SubElement(et_obj, "name").text = "faces"
    ET.SubElement(et_obj, "pose").text = "Unspecified"
    ET.SubElement(et_obj, "truncated").text = "0"
    ET.SubElement(et_obj, "difficult").text = "0"
    bndbox = ET.SubElement(et_obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(x)
    ET.SubElement(bndbox, "ymin").text = str(y)
    ET.SubElement(bndbox, "xmax").text = str(x+w)
    ET.SubElement(bndbox, "ymax").text = str(y+h)

    filename = filename.strip().replace(".jpg", ".xml")
    curr_et_obj.write(filename)

    return curr_et_obj
    
def readAndWrite(train_path, bbx_gttxt_path):
    cnt = 0
    with open(bbx_gttxt_path, "r") as f:
        curr_img = ""
        curr_filename = ""
        curr_path = ""
        curr_et_obj = ET.ElementTree()

        img = np.zeros((80, 80))
        for line in f:
            inp = line.split(' ')
            if len(inp) == 1:
                img_path = inp[0]
                img_path = img_path[:-1]
                curr_img = img_path 
                if curr_img.isdigit():
                    continue 
                print("Image: ", cnt)
                
                img = cv2.imread(train_path + "/" + curr_img, 2) # POSIX only 
                curr_filename = img_path.split("/")[1].strip()
                curr_path = os.path.join(train_path, os.path.dirname(curr_img))
                curr_et_obj = newXMLPASCALfile(img.shape[0], img.shape[1],
                                               curr_path, curr_filename)
                cnt += 1
            else:
                inp = [int(i) for i in inp[:-1]]
                x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = inp
                n = max(w, h)
                if invalid == 1 or blur > 0 or n < 50:
                    continue

                filenow = os.path.join(curr_path,  curr_filename)
                curr_et_obj = appendXMLPASCAL(curr_et_obj, x1, y1, w, h, filenow)

                
#-------------------------------------------------#
# Main 
#-------------------------------------------------#
curr_path = os.getcwd() 

#-------------------------------------------------#
# Training data: 9263 items. Why mine is 12879??
# Only copied out 7000++ images? 
#-------------------------------------------------#
train_path = os.path.join(curr_path, "Data", "WIDER_train", "images")
bbx_gttxt_path = os.path.join(curr_path, "Data", "wider_face_split", "wider_face_train_bbx_gt.txt")
if False: 
    readAndWrite(train_path, bbx_gttxt_path)

# Copy to xml folder
xml_folder = os.path.join(curr_path, "Data", "tf_wider_train", "annotations", "xmls")
image_folder = os.path.join(curr_path, "Data", "tf_wider_train", "images")

# Make folder
try:
    os.makedirs(xml_folder)
    os.makedirs(image_folder)
except Exception as e:
    pass

rootdir_glob = train_path + "/**/*" # This will return absolute pass
file_list = [f for f in iglob(rootdir_glob, recursive = True) if os.path.isfile(f)]

train_annotations_index = os.path.join(curr_path, "Data", "tf_wider_train", "annotations",
                                       "train.txt")

if False: 
    with open(train_annotations_index, "a") as indexFile:
        for f in file_list:
            if ".xml" in f:
                print(f)
                copyfile(f, os.path.join(xml_folder, os.path.basename(f)))
                imgfile = f.replace(".xml", ".jpg")
                copyfile(imgfile, os.path.join(image_folder, os.path.basename(imgfile)))
                indexFile.write(os.path.basename(f.replace(".xml", "")) + "\n")
            
#-------------------------------------------------#
# Validation data 1873
#-------------------------------------------------#
val_path = os.path.join(curr_path, "Data", "WIDER_val", "images")
bbx_gttxt_path = os.path.join(curr_path, "Data", "wider_face_split", "wider_face_val_bbx_gt.txt")
if True: 
    readAndWrite(val_path, bbx_gttxt_path)

# Copy to xml folder
xml_folder = os.path.join(curr_path, "Data", "tf_wider_val", "annotations", "xmls")
image_folder = os.path.join(curr_path, "Data", "tf_wider_val", "images")

# Make folder
try:
    os.makedirs(xml_folder)
    os.makedirs(image_folder)
except Exception as e:
    pass

rootdir_glob = val_path + "/**/*" # This will return absolute pass
file_list = [f for f in iglob(rootdir_glob, recursive = True) if os.path.isfile(f)]

val_annotations_index = os.path.join(curr_path, "Data", "tf_wider_val", "annotations",
                                       "val.txt")

with open(val_annotations_index, "a") as indexFile:
    for f in file_list:
        if ".xml" in f:
            print(f)
            copyfile(f, os.path.join(xml_folder, os.path.basename(f)))
            imgfile = f.replace(".xml", ".jpg")
            copyfile(imgfile, os.path.join(image_folder, os.path.basename(imgfile)))
            indexFile.write(os.path.basename(f.replace(".xml", "")) + "\n")
