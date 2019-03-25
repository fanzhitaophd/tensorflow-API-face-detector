"""
Script to download data from Google Drive using Python 3.6
 - Wider Face training images
 - Wider Face validation images

Wider Face: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/ 

Wider_face_split is included in repo

Ref
 - https://stackoverflow.com/a/16664766

"""

import os 

def download_file_from



#---------------------------------------------------#
# Main 
#---------------------------------------------------#

# Data path
# curr_path = os.getcwd() 
# model_path = os.path.join(curr_path, "Data") 
model_path = "/var/www/Database/Database_face/WIDERFace"

# Make data folder
try:
    os.makedirs(model_path)
except Exception as e:
    pass 

if os.path.exists(os.path.join(model_path, "train.zip")) == false:
    print("Downloading ... train.zip -- 1.47GB")
    download_file_from_google_drive("0B6eKvaijfFUDQUUwd21EckhUbWs", os.path.join(model_path, "train.zip"))

