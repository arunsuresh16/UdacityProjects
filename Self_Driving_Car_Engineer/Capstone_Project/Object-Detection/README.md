# Object-Detection

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

* install COCO API
```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/object_detection/
```

* install pycocotools to site-packages and resolve the import error
```python
python [PATH]/PythonAPI/setup.py build_ext install
```

* install protobuf-compiler and run the compilation process 
```bash
brew install protobuf
protoc [PATH]/Tensorflow/models/object_detection/protos/*.proto --python_out=.
```

* Add libraries to PYTHONPATH
```bash
# From tensorflow/models/object_detection/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### Step3: Install TensorFlow Models Installation
* installing prerequisites in virtualEnv
```bash
conda install pillow, lxml, jupyter, matplotlib, opencv, cython
```
* Creating a `TensorFlow` folder under the root directory of this repo. <br/>
  Then use Git to clone the [TensorFlow Models repo](https://github.com/tensorflow/models) inside `Tensorflow` folder
* git checkout 289a2f99a7df528f6193a5ab3ee284ff3112b731
* Adding necessary environment variables
```python
# cd into TensorFlow/models/object_detection directory
python [PATH]/setup.py build
python [PATH]/setup.py install
```
  
### Step4: generate training and evaluation sets
In order to use TensorFlow Object Detection API, we need to convert raw images into the TFRecord file format first.
Before running the code, Let's create a Training folder under the root directory of the Project.
* run the generate_train_tfrecords.py file and create a train.record file
```python
python [PATH]/generate_train_tfrecords.py --output_path='[PATH]/Object-Detection/Training/train.record'
```

* run the generate_eval_tfrecords.py file and create a eval.record file
```python
python [PATH]/generate_eval_tfrecords.py --output_path='[PATH]/Object-Detection/Training/eval.record'
```

### Step5: Create a label map
Check the label_map.pbtxt under Training directory

### Step6: Create a training configuration
We'll use ssd_mobilenet for experiment, first grab the ssd_mobilenet_v1_coco.config file undeer Tensorflow Project and paste it under Training folder. <br/>
Modify some variables as follows:
* PATH_TO_BE_CONFIGURED: change them to corresponding path to your tfrecords files and label_map file.
* model.ssd.num_classes: number of classes as you defined in label_map.pbtxt
* eval_config.num_examples: number of evalutaion datasets
* fine_tune_checkpoint: the path to the checkpoint file of pretrained model you downloaded. Note: <br/>
    * create a ckpt folder under the root directory of this project and put all the three files you downloaded under it which include .data, .index and .meta files. <br/>
      The reason why we create a separate folder to store checkpoints is because after you retrain the model, new checkpoint files will be generated under Training model and if you put existing checkpoint files under it, it will get confused.
    * although the .data file has suffix like -00000-of-00001. The value of fine_tune_checkpoint should be: [PATH]/Object-Detection/ckpt/model.ckpt without any suffix.

### Step7: Retrain the model
Tensorflow Object Dectection API already has a script to train the model, we just need to run that python script and provide some parameters. <br/>
```bash
python [PATH]/TensorFlow/models/object_detection/train.py --logtostderr --pipeline_config_path=[PATH]/Object-Detection/Training/ssd_mobilenet_v1_coco.config --train_dir=Training/
```
After finish retraining the model, new checkpoint files will be generated under Training directory. <br/>

Similarly, you can evaluate the model using this command: <br/>
```bash
python [PATH]/TensorFlow/models/object_detection/eval.py --logtostderr --checkpoint_dir=Training/ --eval_dir=Training/Evaluation/
```

### Step8: Export inference graph
Choose the checkpoint file with largest index you generated in above step, and run this command:
```bash
python [PATH]/Object-Detection/TensorFlow/models/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path Training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix Training/model.ckpt-317(e.g 317 is the largest index) --output_directory Training/inference_graph
```
Note: the checkpoint filename you provide should be something like this: model.ckpt-317, don't include the remaining suffix (e.g.  .data-00000-of-00001)
After you run this command, an inference_graph folder will be generated under Training folder.