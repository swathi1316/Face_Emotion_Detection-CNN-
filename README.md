PROJECT NAME: EMOTION DETECTION
AUTHORS : SWATHI, VAISHNAVI
DATE : 17th April 2022

### PROJECT: EMOTION DETECTION
**MOTIVATION**
* Few questions your emotion classification model can answer based on your users review
  * What is the sentiment of yours?
  * What is the mood of today's?
* We would like to work on an emotion detection model using Keras and convolution neural networks, We are also using OpenCv to test our model.
* DEEP LEARNING: Deep learning is the machine learning technique that teaches computers to do Naturally to humans: learn by example.

**DATASET DESCRIPTION**
The dataset we are using is from kaggle https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset which is about 126MB. There are about 7 classes namely `sad`,  `happy`, `surprise`, `fear`, `neutral`, `disgust`, `angry`. 
 
**DATA PREPARATION**
* Dataset was uploaded to sagemaker locally as a 'zip' folder and extracted in Jupyter Notebook using unzip command.
* Path of the dataset folder is assigned to a variable and by `listdir` tried to assign the path of the train folder to `train_dir`.
* Path of the dataset folder is assigned to a variable and by `listdir` tried to assign the path of the test folder to `test_dir`.
* To check the images , i randomly selected images with the help of `glob.glob` module by mentioning specific pattern.
* Calculated the length of train and test folder set by using length function on the `listdir`.
* To Balance the data in the classes , Data Augmentation was perfomed by using `ImageDataGenerator` to prevent biased output. Training set was split as 80:20 (train : validation).Same process was implemented on Test data.

**MODEL BUILDING**
* A Sequential model was created.
* Convolutional Neural Networks Conv2D layers are  added inside the sequential model with filters , padding , activation function(`RELU`) and the input shape.
* `MaxPooling` is used to extract the best features from the output of Conv2D which helps in reducing Overfitting.
* `Dropout` is used to manage and balance the accuracy between train and validation sets by increasing or decreasing probability.
* Flatten is used to convert the output of before layer to 1-D Array.
* 2 Fully Connected Layers are used and One output layer with activation layer as `softmax`.
* `ModelCheckPoint` is used to save the model at the end of every Epoch and monitor the performance according to mentioned variable
* `EarlyStopping` is used to stop the job to run when there is no improvement in the parameter pass to EarlyStopping Callback.
* `ReduceLOROnPlateau` helps in monitoring the learning rate (maximizes or minimizes or no improvement).

**TRAINING N TESTING**
* By using `.fit` , trained the training set with the sequential model and mentioned validation set during for `validation_accuracy` and `validation_loss`.
* Mentioned the epochs and the batch size.
* By using `.evaluate` , tested the accuracy and loss of testing data with sequential model.
* Generated a Confusion Matrix by using `model.predict(test_data)` and data visualization was provided for Confusion Matrix.

**Explanation:**
Our classifier is very picky, and does not think many things are hot dogs. All the images it thinks are “happy”, are really “happy”. However it also misses a lot of actual “happy”, because it is so very picky. We have low recall, but a very high precision.

**Run Flask app on local system:**
Flask app for local development is using the flask run command from a terminal. By default, Flask will run the application you defined in app.py on port 5000. While the application is running, go to http://localhost:5000 using your web browser. As we are in the development stage, To reload your application automatically whenever you make a change to it. You can do this by passing an environment variable, FLASK_ENV=development, to flask run.

**Heroku Deployment**
* Login To heroku account, install heroku CLI.
* create a file named Procfile in the project’s root directory. This file tells Heroku how to run the app.
* install Gunicorn and update the requirements.txt file using pip
heroku create face-emo-apps command initializes the Heroku application
* git push heroku master pushing the master branch to the heroku remote
* heroku open will open your application using your default web browser. 

