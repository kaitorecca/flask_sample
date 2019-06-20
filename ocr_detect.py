from urllib.request import urlopen
import numpy as np
import sys
import cv2
import os
from PIL import Image
import pytesseract
import math
import json
import re
import datetime
import urllib
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

def recognize_by_cor(image_path, x, y, height, width):
    image_path = 'http://127.0.0.1:5500/images/1.jpg';
    resp = urlopen(image_path)
    print("Duong dan la: " + image_path)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    src = cv2.imdecode(image, cv2.IMREAD_COLOR)


    # if src is None:
    #     return -1

    new_img = src[y:y+height,x:x+width]
    result = pytesseract.image_to_string(new_img)

    return result

def bounding_box():
    from pytesseract import Output
    img = cv2.imread('python/1.jpg')

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

def rect_img():
    img = cv2.imread('python/2.jpg')

    with open('python/result.json') as f_json:
        data = json.load(f_json)
        for d in data["message"][0]["annotion"]:
            cv2.rectangle(img, (d["anchor_x"], d["anchor_y"]), (d["anchor_x"]+d["anchor_width"],d["anchor_y"]+d["anchor_height"]), (0,123,60),2)

    cv2.imwrite("python/abc.jpg", img)

 
