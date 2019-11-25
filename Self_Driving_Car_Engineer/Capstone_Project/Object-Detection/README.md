# Object-Detection

![Detection](videos/Test1.gif)

### Step1: download datasets

Udacity has some annotated driving [datasets](https://github.com/udacity/self-driving-car/tree/master/annotations) we can use to retrain the existing model. <br/>
You can download the dataset and copy the whole directory `object-dataset` under the root directory of this project.


### Step2: Install all the dependencies
* create a virtualEnv called tensorflow_cpu
```bash
conda create -n tensorflow_cpu pip python=3.6
```

* activate the virtualEnv
```bash
activate tensorflow_cpu
```

* install tensorflow CPU for python
```bash
pip install --ignore-installed --upgrade tensorflow==1.3.0
```

* install protobuf-compiler and run the compilation process 
```bash
brew install protobuf
```

### Step3: Install TensorFlow Models Installation
* installing prerequisites in virtualEnv
```bash
conda install pillow, lxml, jupyter, matplotlib, opencv, cython
```
* Creating a `TensorFlow` folder under the root directory of this repo. <br/>
  Then use Git to clone the [TensorFlow Models repo](https://github.com/tensorflow/models) inside `Tensorflow` folder
* git checkout 289a2f99a7df528f6193a5ab3ee284ff3112b731
* Build the protobuf files
```python
# cd into Tensorflow/models/object_detection directory
protoc protos/*.proto --python_out=.
```
* Adding necessary environment variables
```python
# In the same object_detection directory
python setup.py build
python setup.py install
```
* Add the built libraries to PYTHONPATH
```bash
# In the same object_detection directory
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### Step4: generate training and evaluation sets
In order to use TensorFlow Object Detection API, we need to convert raw images into the TFRecord file format first.

* run the generate_train_tfrecords.py file and create a train.record file
```python
# From the root of the repository
python generate_train_tfrecords.py --output_path='Training/train.record'
```

* run the generate_eval_tfrecords.py file and create an eval.record file
```python
python generate_eval_tfrecords.py --output_path='Training/eval.record'
```

### Step5: Update the label map
Update the label_map.pbtxt inside `Training` directory based on the labels of your choice. 

### Step6: Create a training configuration
We'll use ssd_mobilenet for this experiment. It is already placed inside `Training` directory. 
Modify some variables as follows:
* PATH_TO_BE_CONFIGURED: change them to corresponding path to your tfrecords files and label_map file.
* model.ssd.num_classes: number of classes as you defined in label_map.pbtxt
* eval_config.num_examples: number of evalutaion datasets
* fine_tune_checkpoint: the path to the checkpoint file of pretrained model you downloaded.   
Note:
    * A checkpoint `ckpt` folder is already created inside `Training` directory where the .data, .index and .meta files downloaded from this [link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) are already placed. In case, you want to follow a different model, correspoding model's checkpoint files can be downloaded [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).    
     * The reason why we create a separate folder to store checkpoints is because after we retrain the model, new checkpoint files will be generated under Training model and if you put existing checkpoint files under it, it will get confused.
    * although the .data file has suffix like -00000-of-00001. The value of fine_tune_checkpoint should be: [PATH]/Object-Detection/ckpt/model.ckpt without any suffix.

### Step7: Retrain the model
Tensorflow Object Dectection API already has a script to train the model, we just need to run that python script and provide some parameters. <br/>
```bash
python TensorFlow/models/object_detection/train.py --logtostderr --pipeline_config_path=Training/ssd_mobilenet_v1_coco.config --train_dir=Training/
```
After retraining the model, new checkpoint files will be generated under `Training` directory. <br/>

Similarly, you can evaluate the model using this command: <br/>
```bash
python TensorFlow/models/object_detection/eval.py --logtostderr --pipeline_config_path=Training/ssd_mobilenet_v1_coco.config --checkpoint_dir=Training/ --eval_dir=Training/Evaluation/
```

### Step8: Export inference graph
Before running the inference graph from the newly trained model, make sure to delete the contents of `Training\inference_graph` directory which already has the generated graph. Choose the checkpoint file with largest index you generated in above step, and run this command:
```bash
# Considering 317 as the largest checkpoint the trainer went till
python TensorFlow/models/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path Training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix Training/model.ckpt-317 --output_directory Training/inference_graph
```
Note: the checkpoint filename you provide should be something like this: model.ckpt-317, don't include the remaining suffix (e.g.  .data-00000-of-00001)
After you run this command, saved_model.pb and frozen_inference_graph.pb will be generated. This can be used to load the trained model and then predict/detect objects.