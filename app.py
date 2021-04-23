import cv2
import darknet
import time
import math
import sys
from threading import Thread
from queue import Queue
from ctypes import pointer, c_int
from collections import deque
import matplotlib.pyplot as plt

single_image_queue = Queue(maxsize=1)
series_queue = deque([(0, 0, 0, 0) for i in range(30)], maxlen=30)

<< << << < HEAD
config_file = '/app/data/yolov4.cfg'
data_file = '/app/data/coco.data'
== == == =
config_file = '/app/cfg/yolov4.cfg'
data_file = '/app/cfg/coco.data'
>>>>>> > f8fe1892fa7b96f5180a2c23d9c38a3e3f800d52
weights = '/data/yolov4.weights'

network, class_names, class_colors = darknet.load_network(
    config_file, data_file, weights
)
width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)

input_path = sys.argv[1]
cap = cv2.VideoCapture(input_path)


thresh = .5
hier_thresh = .5


def series(series_queue):
    plt.xticks(range(30))
    while cap.isOpened():
        time.sleep(5)
        series = list(series_queue)
        series = [(ts, bal) for num, bal, fr, ts in series]
        #print('draw series {} ...'.format(*zip(*series)))
        plt.plot(*zip(*series))
        plt.savefig("/data/mygraph.png")
    cap.release()


def inference(single_image_queue, series_queue):
    prev_objects = 0
    def sqt(x): return x * x
    while cap.isOpened():
        frame_id, ts, image = single_image_queue.get()  # 비어있는 경우 대기상태
        # prev_time = time.time()
        darknet.predict_image(network, image)
        pnum = pointer(c_int(0))
        detections = darknet.get_network_boxes(network, image.w, image.h,
                                               thresh, hier_thresh, None, 0, pnum, 0)
        num = pnum[0]
        #detections = darknet.detect_image(network, class_names, darknet_image)
        predictions = []
        for j in range(num):
            for idx, label in enumerate(class_names):
                if detections[j].prob[idx] > 0:
                    predictions.append(
                        (label, str(round(detections[j].prob[idx] * 100, 2))))
        if frame_id > 0:
            curr_objects = len(predictions)
            balance = curr_objects - prev_objects
            prev_objects = curr_objects
            ts = int(ts / 1000)
            frame_id = int(frame_id)
            print('objects: {:<4} diff: {:<4} frame: {:<6} sec: {:<3}'.format(
                curr_objects, sqt(balance), frame_id, ts))
            series_queue.append((curr_objects, sqt(balance), frame_id, ts))

        # fps = int(1/(time.time() - prev_time))
        #fps = 1/(time.time() - prev_time)
        # print("FPS: {}".format(fps))
        darknet.free_detections(detections, num)
        darknet.free_image(image)
    cap.release()


def video_capture(single_image_queue):
    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % math.floor(frameRate) == 0:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            img_for_detect = darknet.make_image(width, height, 3)
            darknet.copy_image_from_bytes(
                img_for_detect, frame_resized.tobytes())
            single_image_queue.put((frame_id, ts, img_for_detect))
    cap.release()


Thread(target=video_capture, args=(single_image_queue,)).start()
Thread(target=inference, args=(single_image_queue, series_queue)).start()
Thread(target=series, args=(series_queue,)).start()
