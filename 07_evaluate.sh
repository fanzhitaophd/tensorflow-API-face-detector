
echo "Did you update the tensorflow_models path in this script?"

cd /home/zhitao/Documents/Python/tensorflow_models/research/object_detection
config_path=/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/ssd_mobilenet_v1_face.config
model_output=/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/model_output  
val=/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/val

python legacy/eval.py --logtostderr \
       --pipeline_config_path=$config_path \
       --checkpoint_dir=$model_output \
       --eval_dir=$val
