"""

Ref
- https://github.com/qdraw/tensorflow-object-detection-tutorial/blob/master/image_object_detection.py
"""

from __future__ import print_function

import os
import cv2
import sys 
import tarfile
import numpy as np 
import tensorflow as tf 
import six.moves.urllib as urllib

from PIL import Image

sys.path.append("/home/zhitao/Documents/Python/tensorflow_models/research/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util

#---------------------------------------------------#
# subfunction 
#---------------------------------------------------#
def load_image_into_numpy_array(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((h,w,3)).astype(np.uint8)

#---------------------------------------------------#
# main 
#---------------------------------------------------#
# Path to frozen detection graph
# MODEL_NAME = "/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/model_yeephycho"
MODEL_NAME = "/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/model"
PATH_TO_CKPT = MODEL_NAME + "/frozen_inference_graph.pb"
PATH_TO_LABEL = "/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/model/face_label_map.pdtxt"

NUM_CLASSES = 1

# Load model to memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

PATH_TEST_IMAGE_DIR = 'Test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TEST_IMAGE_DIR, 'image{}.jpg'.format(i)) for i in range(3,7)]

IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict = {image_tensor: image_np_expanded}
            )

            # Visualize detected boxes
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8
            )

            # Display 
            print('Image '+ image_path.split('.')[0] + '_labeled.jpg')
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            cv2.imshow("Detection", image_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
