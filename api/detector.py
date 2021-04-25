from flask import Blueprint, render_template, Response
import os
import cv2
import darknet
import json
import math

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
    darknet_image = darknet.make_image(width, height, 3)

    input_path = '/data/test10.mp4'
    cap = cv2.VideoCapture(input_path)
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
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # img_for_detect = darknet.make_image(width, height, 3)
            # darknet.copy_image_from_bytes(
            #     img_for_detect, frame_resized.tobytes())
            # single_image_queue.put((frame_id, ts, img_for_detect))
    cap.release()
    return json.dumps(os.path.exists(input_path))