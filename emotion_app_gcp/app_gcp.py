from importlib.resources import contents
from unittest import result
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import io
from google.cloud import vision
from google.oauth2 import service_account

# Initialise Flask
app = Flask(__name__)

# Provide credentials to authenticate to a Google Cloud API
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'smooth-command-342916-483c405ae2e0.json'
key_path = './smooth-command-342916-483c405ae2e0.json'
credentials = service_account.Credentials.from_service_account_file(key_path)



import base64

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict


def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    credentials = credentials
    
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options, credentials=credentials)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))
    return predictions

# Temporary storage for uploaded pictures
# to be able to display uploaded pictures they should locate in static directory
UPLOAD_FOLDER = 'static/tmp' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Maximum Image Uploading size
# app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

# Image extension allowed
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods = ['POST', 'GET'])
def upload_file():
   if request.method == 'POST':
      file = request.files['file']
      if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        content = predict_image_classification_sample(
                 project="375688213202",
                endpoint_id="6671493509716901888",
                location="us-central1",
                 filename=filepath ) 
        for content in content:
            print(" prediction:", dict(content))
            return render_template("search_results.html", content=content, original=filepath)

def search(f):
    import io
    from google.cloud import vision
  
    client = vision.ImageAnnotatorClient()

    with io.open(f, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection
    
    best_guess_label = None
    if annotations.best_guess_labels:
        labels = [label.label for label in annotations.best_guess_labels]
        best_guess_label = ','.join(labels)

    web_entities = []
    if annotations.web_entities:
        web_entities = annotations.web_entities

    visually_similar_images = []
    if annotations.visually_similar_images:
        visually_similar_images = [image.url for image in annotations.visually_similar_images]

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return best_guess_label, web_entities, visually_similar_images

if __name__ == '__main__':
    app.run(debug=True)
