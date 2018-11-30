import base64
import numpy as np 
import io
from PIL import Image
import keras
from keras import backend as K 
#from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.preprocessing import image 
from flask import request
from flask import jsonify
from flask import Flask 
import cv2 
import re

app = Flask(__name__)

def get_model():
	global model
	model = load_model('models/my_model_flask.h5')
	print(" * Model loaded!")

def preprocess_image(image, target_size):
	if image.mode !="RGB":
		image = image.convert("RGB")
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	return image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

with open ("dog_names.txt", "r") as myfile:
    dog_names = list(map(lambda x: x.strip(),myfile.readlines()))

print(" *Loading Keras model...") 
get_model() #load model into memory

def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


@app.route("/predict", methods=["POST"])

def predict():
	message = request.get_json(forse=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image, (224,224))
	bottleneck_feature = extract_Xception(processed_image)
	predicted_vector = model.predict(bottleneck_feature)
	breed, confidence = dog_names[np.argmax(predicted_vector)].split(".")[1], int(round(100*np.max(predicted_vector))) #tuple to list?
	prediction=breed, confidence
	response = {
		'prediction': {
			'breed': prediction[0],
			'confidence':prediction[1]
		}
	} 
	return jsonify(response)