### Docker 실행

```bash
docker run -it --rm -v `pwd`:/data -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 yolo4 python3 /data/detector.py /data/test10.mp4

docker run -it --rm -v "%cd%":/data -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 yolo4 python3 /data/detector.py /data/test10.mp4

docker run -it --rm -v `pwd`:/data -p 5000:5000 -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 -e PYTHONPATH="/app:/data" yolo4 python3 /data/app.py
```

### yolov4.weights 주소

https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
