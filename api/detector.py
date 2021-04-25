from flask import Blueprint, render_template, Response
import os
import cv2
import darknet
import json
import math
from ctypes import pointer, c_int

detector = Blueprint("detector", __name__, static_folder="static", template_folder="template")

@detector.route("/")
def home():
    return render_template("home.html")

@detector.route("/detect")
def detect():
    config_file = '/app/cfg/yolov4.cfg'
    data_file = '/app/cfg/coco.data'
    weights = '/data/yolov4.weights'

    network, class_names, class_colors = darknet.load_network(
        config_file, data_file, weights
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    input_path = '/data/test10.mp4'

    return Response(video_feed(input_path, network, class_names,
class_colors, width, height),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

def video_feed(input_path, network, class_names, class_colors, width, height):
    cap = cv2.VideoCapture(input_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    thresh = .5
    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % math.floor(frame_rate) == 0:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                interpolation=cv2.INTER_LINEAR)
            image = darknet.make_image(width, height, 3)
            darknet.copy_image_from_bytes(image, frame_resized.tobytes())
            detections = darknet.detect_image(network, class_names, image, thresh=thresh)
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            (flag, encodedImage) = cv2.imencode(".jpg", image)
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

            # darknet.free_detections(detections, num)
            # darknet.free_image(image)
            # img_for_detect = darknet.make_image(width, height, 3)
            # darknet.copy_image_from_bytes(
            #     img_for_detect, frame_resized.tobytes())
            # single_image_queue.put((frame_id, ts, img_for_detect))
    cap.release()