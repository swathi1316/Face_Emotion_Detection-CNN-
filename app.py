from flask import Flask, render_template, request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import gunicorn

app = Flask(__name__)

emotion_dict = {0:'angry', 1:'neutral',2:'sad'}

model = load_model('face_emotion_e.h5')

model.make_predict_function()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_label(img_path):
	img = image.load_img(img_path, target_size=(48,48))
	img = image.img_to_array(img)/255.0
	img = img.reshape(3, 48,48,1)
	prediction = model.predict(img)
	prediction = np.argmax(prediction,axis=0)
	pred = max(prediction)
	p = np.where(prediction == pred)
	pred = p[0]
	return emotion_dict[int(pred)]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		if img and allowed_file(img.filename):
			img_path = "static/" + img.filename	
			img.save(img_path)
			img = cv2.imread(img_path)
			p = predict_label(img_path)
			return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)