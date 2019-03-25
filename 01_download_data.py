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
import requests

def download_file_from_google_drive(id, dest): 
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            return value
        return None

    def save_response_content(response, dest):
        CHUNK_SIZE = 32768
        with open(dest, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "http://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params = {'id' : id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id' : id, 'confirm' : token}
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, dest)
        
#---------------------------------------------------#
# Main 
#---------------------------------------------------#

# Data path
curr_path = os.getcwd() 
model_path = os.path.join(curr_path, "Data") 
# model_path = "/var/www/Database/Database_face/WIDERFace"

# Make data folder
try:
    os.makedirs(model_path)
except Exception as e:
    pass 

if os.path.exists(os.path.join(model_path, "train.zip")) == False:
    print("Downloading ... train.zip -- 1.47GB")
    download_file_from_google_drive("0B6eKvaijfFUDQUUwd21EckhUbWs", os.path.join(model_path, "train.zip"))

if os.path.exists(os.path.join(model_path, "val.zip")) == False:
    print("Downloading ... val.zip -- 362.8MB")
    download_file_from_google_drive("0B6eKvaijfFUDd3dIRmpvSk8tLUk", os.path.join(model_path, "val.zip"))

    
