# TF Lite prediction without TF/Keras dependencies

from PIL import Image
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from urllib import request
from io import BytesIO
import numpy as np



classes = ['dino', 'dragon']

def preprocess_input(x):
    x /= 255
    return x

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# Open the TF Lite model
tflite_interpreter = tflite.Interpreter("dino-vs-dragon-v2.tflite")
input_index = tflite_interpreter.get_input_details()[0]["index"]
output_index = tflite_interpreter.get_output_details()[0]["index"]

def predict(url):
    # Get image from URL
    img = download_image(url)
    img = prepare_image(img, (150,150))
    x = np.array(img, dtype=np.float32)
    img = preprocess_input(x)
    X = np.array([x])

    # Allocate tensors
    tflite_interpreter.allocate_tensors()
    # Set input to the TF Lite model
    tflite_interpreter.set_tensor(input_index, X)
    # Predict
    tflite_interpreter.invoke()
    # Get prediction output
    pred = tflite_interpreter.get_tensor(output_index)
    pred = pred.flatten().tolist()

    return dict(zip([classes[np.round(prediction).astype(np.uint8)] for prediction in pred], pred))

def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return 
