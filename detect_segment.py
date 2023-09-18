import os
import sys
import argparse
import logging
import time
from pathlib import Path
import glob
import json
from edgetpumodel_segmentation import EdgeTPUSegmentation
import rospy
import numpy as np
from tqdm import tqdm
import cv2
import yaml
import timeit
from collections import deque
import matplotlib.pyplot as plt

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from edgetpumodel_change import EdgeTPUModel
from utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class, StreamingDataProcessor

dist_save = np.array([])
ema_distance_list = []
max_circle_queue_size = 20 
detected_circles_queue = deque(maxlen=max_circle_queue_size)

class priROS:
    def __init__(self):
        rospy.init_node('kudos_vision', anonymous = False)
        self.yolo_result_img_pub = rospy.Publisher("/output/image_raw2/compressed", CompressedImage, queue_size = 1)
        self.distance_pub = rospy.Publisher("/distance_topic", Float64, queue_size=1)  # Adjust topic name and message type

    def yolo_result_img_talker(self, image_np,fps):
        import cv2
        # print("Mean FPS: {:1.2f}".format(fps))
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tobytes()
        self.yolo_result_img_pub.publish(msg)

    def distance_talker(self, distance):
        # Publish ema_distance as a Float64 message
        # print(Float64(distance))
        self.distance_pub.publish(Float64(distance))  # Adjust message type if needed

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", help="weights file", required=True)
    parser.add_argument("--bench_speed", action='store_true', help="run speed test on dummy data")
    parser.add_argument("--bench_image", action='store_true', help="run detection test")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--names", type=str, default='data/coco.yaml', help="Names file")
    parser.add_argument("--image", "-i", type=str, help="Image file to run detection on")
    parser.add_argument("--device", type=int, default=2, help="Image capture device to run live detection")
    # Device num : v4l2-ctl --list-devices
    parser.add_argument("--stream", action='store_true', help="Process a stream")
    parser.add_argument("--bench_coco", action='store_true', help="Process a stream")
    parser.add_argument("--coco_path", type=str, help="Path to COCO 2017 Val folder")
    parser.add_argument("--quiet","-q", action='store_true', help="Disable logging (except errors)")
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='(optional) dataset.yaml path') 

    args = parser.parse_args()
    priROS = priROS()
    if args.quiet:
        logging.disable(logging.CRITICAL)
        logger.disabled = True
    
    if args.stream and args.image:
        logger.error("Please select either an input image or a stream")
        exit(1)
    desired_width = 640
    desired_height = 480

    segmentation_model = EdgeTPUSegmentation(args.segmentation_model, desired_width, desired_height)
    
    conf_thresh = 0.25
    iou_thresh = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000

    ema_distance = None

    if args.stream:
        logger.info("Opening stream on device: {}".format(args.device))
        total_times = []
        constant = 40800
        count = 0

        
        cam = cv2.VideoCapture(args.device)
        
        # Set the camera frame size
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
        
        start_time = time.time()  # Record the start time

        # Create a StreamingDataProcessor instance
        data_processor = StreamingDataProcessor(window_size=10, alpha=0.2, z_threshold=1)

        while True:

            try:
                res, image = cam.read()
                height, width = image.shape[:2]
                # new_img_size = (width, width)
                if res is False:
                    logger.error("Empty image received")
                    break
                else:
                    
                    # cameraMatrix 정의
                    cameraMatrix = np.array([[9.4088597780774421e+02, 0., 3.7770158111216648e+02],
                                            [0., 9.4925081349933703e+02, 3.2918621409818121e+02],
                                            [0., 0., 1.]])

                    # distCoeffs 정의
                    distCoeffs = np.array([-4.4977607383629226e-01, -3.0529616557684319e-01,
                                        -3.9021603448837856e-03, -2.8130335366792153e-03,
                                        1.2224960045867554e+00])
                    image = cv2.undistort(image, cameraMatrix, distCoeffs) #, None, new_img_size)

                    total_times = []
                    
                    # Segment the image using the segmentation_model
                    segmentation_result = segmentation_model.segment_image(image)
                    
                    # Calculate FPS and publish the result image
                    total_times = np.append(total_times, tinference + tnms)
                    total_times = np.array(total_times)
                    fps = 1.0 / total_times.mean()
                    
                    # Publish segmented image
                    priROS.yolo_result_img_talker(segmentation_result, fps)
                    
                    tinference, tnms = segmentation_model.get_last_inference_time()  # Use segmentation_model
                    logger.info("Frame done in {}".format(tinference+tnms))
                                    
            except KeyboardInterrupt:
                cam.release()
                pass
            except Exception as e:
                print(e)
                pass

        cam.release()