import cv2
import darknet
import time
import math
import sys
from threading import Thread, enumerate
from queue import Queue

darknet_image_queue = Queue(maxsize=1)

config_file = '/app/data/yolov4.cfg'
data_file = '/app/data/coco.data'
weights = '/data/yolov4.weights'

network, class_names, class_colors = darknet.load_network(
    config_file, data_file, weights
)
width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)

input_path = sys.argv[1]
cap = cv2.VideoCapture(input_path)

frameRate = cap.get(cv2.CAP_PROP_FPS)
print('frame rate is {}'.format(frameRate))


def inference(darknet_image_queue):
    prev_objects = 0
    while cap.isOpened():
        frameId, ts, darknet_image = darknet_image_queue.get()  # 비어있는 경우 대기상태
        # prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image)
        delta_objects = len(detections) - prev_objects
        delta_objects = delta_objects * delta_objects
        prev_objects = len(detections)
        ts = int(ts / 1000)
        frameId = int(frameId)
        print('objects: {:<4} diff: {:<3} frame: {:04d} sec: {:04d}'.format(
            len(detections), delta_objects, frameId, ts))
        # fps = int(1/(time.time() - prev_time))
        # fps = 1/(time.time() - prev_time)
        # print("FPS: {}".format(fps))
        darknet.free_image(darknet_image)
    cap.release()


def video_capture(darknet_image_queue):
    while cap.isOpened():
        frameId = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            img_for_detect = darknet.make_image(width, height, 3)
            darknet.copy_image_from_bytes(
                img_for_detect, frame_resized.tobytes())
            darknet_image_queue.put((frameId, ts, img_for_detect))
    cap.release()


Thread(target=video_capture, args=(darknet_image_queue,)).start()
Thread(target=inference, args=(darknet_image_queue,)).start()
