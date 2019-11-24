import tensorflow as tf
from object_detection.utils import dataset_util
import csv
import os

base_path = os.getcwd()
flags = tf.app.flags
flags.DEFINE_string('output_path', '', '{}/Training/eval.record'.format(base_path))
FLAGS = flags.FLAGS

LABEL_FILE_PATH = "{}/object-dataset/labels.csv".format(base_path)
LABEL_DICT = {
    "Green": 1,
    "Yellow": 2,
    "Red": 3
}

def create_tf_example(row):

    filename = row[0]
    xmin = row[1]
    ymin = row[2]
    xmax = row[3]
    ymax = row[4]
    class_text = row[7]

    # datasets include these different labels: Green, GreenLeft, YellowLeft, Yellow, RedLeft, Red
    # we only need three classes: Red, Yellow, Green  or maybe two classes: Red, Not Red
    if class_text.startswith('Red'):
        class_text = 'Red'
    elif class_text.startswith('Yellow'):
        class_text = 'Yellow'
    else:
        class_text = 'Green'

    img_path = "{}/object-dataset/{}".format(base_path, filename)

    # open the image
    with open(img_path, "rb") as f:
        encoded_image_data = f.read()
        height = 1200  # Image height
        width = 1920  # Image width
        filename = filename.encode()
        image_format = b'jpg'
        xmins = [int(xmin) / width]
        ymins = [int(ymin) / height]
        xmaxs = [int(xmax) / width]
        ymaxs = [int(ymax) / height]
        classes_text = [class_text.encode()]
        classes = [LABEL_DICT[class_text]]

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    with open(LABEL_FILE_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')

        # There are 14693 images with traffic light label
        # Use first 10285 images (around 70%) as training set
        # remaining 4408 will be used as evaluation set
        counter = 0
        eval_counter = 0
        for row in csv_reader:
            # e.g. ['1478901536388465963.jpg', '1048', '552', '1348', '748', '0', 'car']

            # only process images with traffic light label
            label = row[6]
            if label != "trafficLight":
                continue
            try:
                attributes = row[7]
            except Exception as e:# some traffic light doesn't have attributes
                continue

            counter += 1
            if counter < 10285:
                continue

            eval_counter += 1
            # convert img to tf_example
            tf_example = create_tf_example(row)
            writer.write(tf_example.SerializeToString())
        print("{} images are used as evaluation sets".format(eval_counter))

    writer.close()

if __name__ == "__main__":
    tf.app.run()
