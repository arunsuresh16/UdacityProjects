import os
import cv2
import numpy as np
import rospy
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor

import matplotlib.pyplot as plt

from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self, model_file):
        #TODO load classifier
        self.current_light = TrafficLight.UNKNOWN
        self.enable_visualization = False
        #Get Current directory
        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, "train_model/{}".format(model_file))
        #rospy.logwarn("model_path={}".format(model_path))
        
        # load inference graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        self.category_index = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Yellow'},
                               3: {'id': 3, 'name': 'Red'}}
        #Configure TF session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        #Create Session
        self.sess = tf.Session(graph=self.detection_graph, config=config)
        
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def to_image_coords(self, boxes, height, width):
        
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords

    def draw_boxes(self, image, boxes, classes, thickness=6):
        
        """Draw bounding boxes on the image"""

        COLOR_LIST = ['green', 'yellow', 'red']

	height, width, dim = image.shape
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        for i in range(len(boxes)):

            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i]-1)
            color = COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
	image = np.array(image)

	return image
        
    def filter_boxes(self, min_score, boxes, scores, classes):
        
        n = len(classes)
        
        idxs = []
        
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
                
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        
        return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        
        with tf.Session(graph=self.detection_graph) as self.sess:                
            # Actual detection.
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                        feed_dict={self.image_tensor: image_np})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            confidence_cutoff = 0.1
        
            # Filter boxes with a confidence score less than `confidence_cutoff`
            final_boxes, final_scores, final_classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
            #print(final_classes, final_scores)
	    #print(final_classes.size)

            # Enable for Visualization
	    if self.enable_visualization and len(final_boxes) > 0 :
		width, height, channels = image.shape
            	box_coords = self.to_image_coords(final_boxes, height, width)
            	image = self.draw_boxes(image, box_coords, final_classes)
	    	#plt.figure(figsize=(12, 8))
            	#plt.imshow(image)
            
        
        if final_classes.size != 0 and final_scores[0] is not None and final_scores[0] > confidence_cutoff:
            class_name = self.category_index[final_classes[0]]['name']
            #print((class_name, final_scores[0]))
            #rospy.logwarn("final_class_prediction={}".format(final_classes[0]))
            if final_classes[0] == 1:
                return TrafficLight.GREEN
            elif final_classes[0] == 2:
                return TrafficLight.YELLOW
            elif final_classes[0] == 3:
                return TrafficLight.RED
            
        
        return TrafficLight.UNKNOWN

    
