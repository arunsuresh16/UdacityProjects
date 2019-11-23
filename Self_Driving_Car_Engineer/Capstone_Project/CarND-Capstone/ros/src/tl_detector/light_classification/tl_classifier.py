import os
import cv2
import numpy as np
import rospy
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from styx_msgs.msg import TrafficLight

SAVE_DETECTED_IMAGES = 0
CONFIDENCE_CUTOFF = 0.13
COLOR_LIST = ['green', 'yellow', 'red']
TRAFFICLIGHT_LIST = [TrafficLight.GREEN, TrafficLight.YELLOW, TrafficLight.RED]


class TLClassifier(object):
    def __init__(self, model_file):
        self.current_light = TrafficLight.UNKNOWN

        #Get Current directory
        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, "train_model/{}".format(model_file))
        rospy.logdebug("model_path={}".format(model_path))
        
        # load inference graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #Configure TF session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        #Create Session
        self.sess = tf.Session(graph=self.detection_graph, config=config)
        
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent the level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.light_state_changed = True
        self.prev_class_id = 0
        self.saved_image_count = 0

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
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        # rospy.logwarn("Found boxes {0}, classes {1} for count {2}".format(boxes, classes, self.saved_image_count))

        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])-1
            color = COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

        self.saved_image_count += 1
        image_name = "images/saved_image_with_box" + str(self.saved_image_count) + ".jpg"
        image.save(image_name)

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
                
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def find_state_change_and_noise_filter(self, boxes, classes):
        flag = False
        filtered_boxes = []
        new_classes = []

        # class_count is defined as per COLOR_LIST. This is required to count
        # which colored box has been detected more and then ignore the less dominant colored box.
        class_count = [0, 0, 0]
        for i in range(len(classes)):
            class_count[int(classes[i]) - 1] += 1

        # rospy.logwarn("Found classes {}".format(classes))
        new_class_id = class_count.index(max(class_count)) + 1
        # Detect only if there is a change in the state
        if self.prev_class_id is not new_class_id:
            for i in range(len(boxes)):
                if (int(classes[i]) is new_class_id):
                    filtered_boxes.append(boxes[i])
                    new_classes.append(classes[i])
                    self.prev_class_id = new_class_id
                    flag = True
                    self.current_light = TRAFFICLIGHT_LIST[int(new_class_id) - 1]

            if flag:
                color = COLOR_LIST[int(new_class_id)-1]
                rospy.logwarn("{} light detected".format(color))

        return flag, np.array(filtered_boxes), new_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=self.detection_graph) as self.sess:
            # Actual detection.
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                     feed_dict={self.image_tensor: image_np})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
        
            # Filter boxes with a confidence score less than CONFIDENCE_CUTOFF
            final_boxes, final_scores, final_classes = self.filter_boxes(CONFIDENCE_CUTOFF, boxes, scores, classes)

            if len(final_boxes) > 0:
                state, final_boxes, final_classes = self.find_state_change_and_noise_filter(final_boxes, final_classes)

                if SAVE_DETECTED_IMAGES and len(final_boxes) > 0:# and state is True:
                    # Each class will be represented by a differently colored box
                    height, width, channels = image.shape
                    box_coords = self.to_image_coords(final_boxes, height, width)
                    self.draw_boxes(image, box_coords, final_classes)

        return self.current_light
