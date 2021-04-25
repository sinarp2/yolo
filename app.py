import flask
from flask import request, jsonify
from api.detector import detector

import darknet

app = flask.Flask(__name__)
app.register_blueprint(detector, url_prefix='/yolo')

if __name__ == '__main__':
    app.config["DEBUG"] = True
    app.run(host="0.0.0.0", debug=True)