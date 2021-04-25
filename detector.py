import cv2
import darknet
import time
import math
import sys
from threading import Thread
from queue import Queue
from ctypes import pointer, c_int
from collections import deque
# import matplotlib.pyplot as plt

single_image_queue = Queue(maxsize=1)
series_queue = deque([(0, 0, 0, 0) for i in range(30)], maxlen=30)

config_file = '/app/cfg/yolov4.cfg'
data_file = '/app/cfg/coco.data'
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
    # plt.xticks(range(30))
    while cap.isOpened():
        pass
        # print(list(series_queue))
        # time.sleep(5)
        # series = list(series_queue)
        # series = [(ts, bal) for num, bal, fr, ts in series]
        # #print('draw series {} ...'.format(*zip(*series)))
        # plt.plot(*zip(*series))
        # plt.savefig("/data/mygraph.png")
    cap.release()


def inference(single_image_queue, series_queue):
    prev_objects = 0
    def sqt(x): return x * x
    def log(x): return x if x == 0 else math.log(sqt(x))
    while cap.isOpened():
        frame_id, image = single_image_queue.get()  # 비어있는 경우 대기상태
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
        # 이전 객체 개 수와 비교
        curr_objects = len(predictions)
        balance = curr_objects - prev_objects
        prev_objects = curr_objects
        if frame_id == 0:
            # 첫 프레임은 비교 대상이 없으므로 스킵
            continue
        # ts = int(ts / 1000)
        # ts = ts / 1000.0
        frame_id = int(frame_id)
        print('frame: {:<4} objects: {:<4} diff: {:<4}'.format(
            frame_id, curr_objects, log(balance)))
        series_queue.append((curr_objects, log(balance), frame_id))

        # fps = int(1/(time.time() - prev_time))
        #fps = 1/(time.time() - prev_time)
        # print("FPS: {}".format(fps))
        darknet.free_detections(detections, num)
        darknet.free_image(image)
    cap.release()


def video_capture(single_image_queue):
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = frame_rate / 3
    while cap.isOpened():
        # 프레임을 읽기 전 프레임 위치는 -> 0
        # 프레임을 읽고 난 후 프레임 위치는 -> 1
        # if frame_id % math.floor(frame_rate) == 0 조건을 위해  프레임을 0 부터 시작
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % math.floor(frame_rate) == 0:
            # 시간은 의미 없음 frame 넘버와 frame rate으로 시간이 결정됨
            # ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            img_for_detect = darknet.make_image(width, height, 3)
            darknet.copy_image_from_bytes(
                img_for_detect, frame_resized.tobytes())
            single_image_queue.put((frame_id, img_for_detect))
    cap.release()


Thread(target=video_capture, args=(single_image_queue,)).start()
Thread(target=inference, args=(single_image_queue, series_queue)).start()
Thread(target=series, args=(series_queue,)).start()
