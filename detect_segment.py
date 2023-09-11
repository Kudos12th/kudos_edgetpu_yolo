import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import rospy
import numpy as np
from tqdm import tqdm
import cv2
import yaml
from collections import deque
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage
from tflite_runtime.interpreter import Interpreter, load_delegate
import tensorflow as tf
from std_msgs.msg import Float64
from pycoral.utils import edgetpu

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

from PIL import Image



def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap


  
def label_to_color_image(label):
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


# class priROS:
#     def __init__(self):
#         rospy.init_node('kudos_vision', anonymous = False)
#         self.yolo_result_img_pub = rospy.Publisher("/output/image_raw2/compressed", CompressedImage, queue_size = 1)
#         self.distance_pub = rospy.Publisher("/distance_topic", Float64, queue_size=1)  # Adjust topic name and message type

#     def yolo_result_img_talker(self, image_np,fps):
#         # print("Mean FPS: {:1.2f}".format(fps))
#         msg = CompressedImage()
#         msg.header.stamp = rospy.Time.now()
#         msg.format = "jpeg"
#         msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tobytes()
#         self.yolo_result_img_pub.publish(msg)

#     def distance_talker(self, distance):
#         # Publish ema_distance as a Float64 message
#         # print(Float64(distance))
#         self.distance_pub.publish(Float64(distance))  # Adjust message type if needed

# priROS = priROS()

# Set the desired frame size
desired_width = 640
desired_height = 640

w = 'seg_edgetpu.tflite'

cam = cv2.VideoCapture(2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

interpreter = make_interpreter(w, device=':0')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
width, height = common.input_size(interpreter)
while cam.isOpened():
    res, image = cam.read()

    if res is False:
        logger.error("Empty image received")
        break
    else:
        total_times = []

        input_data = cv2.resize(image, (width, height))
        input_data = np.expand_dims(input_data, axis=0)
        input_data_int8 = input_data.astype('int8')

        interpreter.set_tensor(input_details[0]['index'], input_data_int8)
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        result = segment.get_output(interpreter)
        
        print(result)

        segmented_frame = label_to_color_image(result).astype(np.uint8)
        cv2.imshow("Segmented Frame", segmented_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cam.release()
cv2.destroyAllWindows()