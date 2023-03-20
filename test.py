# import YOLO model
from ultralytics import YOLO
# import maths functions
import numpy as np
# import image functions
from PIL import Image
# import requests function that grabs an image from a URL
import requests
# import methods to manipulate bytes data in memory
from io import BytesIO
# import library to perform image processing
import cv2
# import function to read an image from a file into an arry
import matplotlib.pyplot as plt

# save the model
model = YOLO("yolov8n.pt")

# ask the user for the file directory
image_path = str(input('Image directory path: ')).strip()

# convert the image into an array
image = plt.imread(image_path)

# predict the detected objects and output to the user
results = model.predict(image)
