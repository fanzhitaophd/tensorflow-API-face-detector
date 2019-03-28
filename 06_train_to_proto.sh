
echo "Did you update the tensorflow_models path in this script?" 

config_path=/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/ssd_mobilenet_v1_face.config
model_output=/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/model_output  
model=/home/zhitao/Documents/Python/Object_detection/Train_SSD/tensorflow-API-face-detector/model  

cd /home/zhitao/Documents/Python/tensorflow_models/research/object_detection
python export_inference_graph.py --input_type image_tensor \
       --pipeline_config_path=$config_path \
       --trained_checkpoint_prefix=$model_output/model.ckpt-0 \
       --output_directory=$model


