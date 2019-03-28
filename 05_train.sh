
# Go to tensorflow research folder
# python setup.py build
# python setup.py install 
# cd slim 
# sudo pip install -e . 


echo "Did you update the paths ssd_mobilenet_v1_face.config?"
echo "and tensorflow_models in this script"

cd /home/zhitao/Documents/Python/tensorflow_models/research/object_detection
config_path=/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/ssd_mobilenet_v1_face.config
python legacy/train.py --logtostderr --pipeline_config_path=$config_path --train_dir=model_output

