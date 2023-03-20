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

user_pick = str(input('1 for image in URL or 2 for image in directory path: ')).strip()

if user_pick == "1":
    # ask the user for the URL
    image_url = str(input('Image URL: ')).strip()

     # ask the user for a confidence level
    confidence = str(input('Confidence level: ')).strip()
    confidence_level = float(confidence)

    # send a get request to the URL
    response = requests.get(image_url)
    # read and save image data
    image = Image.open(BytesIO(response.content))
    # convert the image into an array
    image = np.asarray(image)
    # predict the detected objects and output to the user
    results = model.predict(image, conf=confidence_level)

elif user_pick == "2":
    # ask the user for the file directory
    image_path = str(input('Image directory path: ')).strip()

    # ask the user for a confidence level
    confidence = str(input('Confidence level: ')).strip()
    confidence_level = float(confidence)

    # predict the detected objects and output to the user
    results = model.predict(source='{}'.format(image_path), conf=confidence_level)
    
else: 
    print("Invalid choice. Please restart programme.")
    quit()
