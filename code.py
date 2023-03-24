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
# import custom dataset from Roboflow
from roboflow import Roboflow

# download and unzip Roboflow dataset
rf = Roboflow(api_key="HAijjYs0jW2f55BBoIS0")
project = rf.workspace("machine-learning-yfysx").project("machine-learning-mooc")
dataset = project.version(6).download("yolov8")

# load the standard model or can be the empty .yaml file since we will be overwriting it with custom data
# model = YOLO("yolov8n.pt")

# load the trained model 
model = YOLO("runs/detect/train31/weights/best.pt")

# train the model using the Roboflow dataset
# results = model.train(data=dataset.location + "/data.yaml", epochs=150)

# save trained model
# success = model.export(format="onnx")

# prompt user
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