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

## Install

This project requires **Python** and the following Python libraries and Heroku CLI installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Math](https://docs.python.org/3/library/math.html)
- [Seaborn](https://seaborn.pydata.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [PIL](https://pillow.readthedocs.io/en/stable/installation.html)
- [Flask](https://pypi.org/project/Flask/)
- [Gunicorn](https://docs.gunicorn.org/en/stable/install.html)
- [Heroku](https://devcenter.heroku.com/articles/heroku-cli)

You will also need to have SageMaker Studio Lab Account activated to run and execute a [Jupyter Notebook](http://jupyter.org/install.html)

## Code

Code is provided in the `face_emotion_detection_m.ipynb` notebook file

### Importing the Dataset
 
* Dataset was uploaded to sagemaker locally as a 'zip' folder and extracted in Jupyter Notebook using unzip command.

### Exploratory Data Exploration

* Path of the dataset folder is assigned to a variable and by `listdir` tried to assign the path of the train folder to `train_dir`.
* Path of the dataset folder is assigned to a variable and by `listdir` tried to assign the path of the test folder to `test_dir`.
* To check the images , i randomly selected images with the help of `glob.glob` module by mentioning specific pattern.
* Calculated the length of train and test folder set by using length function on the `listdir`.
* To Balance the data in the classes , Data Augmentation was perfomed by using `ImageDataGenerator` to prevent biased output. Training set was split as 80:20 (train : validation).Same process was implemented on Test data.
 
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

**Experiments:**

***Experiment 1***
Experimented with two convolutional layers and one FC layer. In the first convolutional layer, we had 32 3×3 filters, with the stride of size 1, along with batch normalization and dropout, but without max pooling. In the second convolutional layer, we had 64 3×3 filters, with the stride of size 1, along with batch normalization and dropout and also max-pooling with a filter size 2×2. In the FC layer, we had a hidden layer with 512 neurons and Softmax as the loss function. Also in all the layers, we used Rectified Linear Unit (ReLU) as the activation function. For the training process, we used all of the images in the training set with 30 epochs and a batch size of 128 and cross-validated the hyper-patameters of the model with different values for regularization, learning rate and the number of hidden neurons . To validate our model in each iteration, we used the validation set and to evaluate the performance of the model, we used the test set. The accuracy is about 40% and validation loss is 0.9. 
To observe the effect of adding convolutional layers and FC layers to the network, we trained a deeper CNN with 4 convolutional layers and two FC layers. The first convolutional layer had 64 3×3 filters, the second one had 128 5×5 filters, the third one had 512 3×3 filters and the last one had 512 3×3 filters. In all the convolutional layers, we have a stride of size 1, batch normalization, dropout, max-pooling and ReLU as the activation function. The hidden layer in the first FC layers had 256 neurons and the second FC layer had 512 neurons. In both FC layers, same as in the convolutional layers, we used batch normalization, dropout and ReLU. Also we used Softmax as our loss function. Figure 2 shows the architecture of this deep network. As in the shallow model, before training the network, we performed initial loss checking and examined the ability of overfitting the network using a small subset of the training set. The results of these sanity checks proved that the implementation of the network was correct. Then, using 35 epochs and a batch size of 128, we trained the network with all the images in the training set. Moreover, we cross-validated the hyperparameters to get the model with the highest accuracy. This time, we obtained loss: 0.7867,  accuracy: 70%, val_loss: 1.3778, val_accuracy: 55%. The difference between accuracy and val_accuracy is comparatively more. 

***Experiment 2:***
Decided to do some analysis using Vertex AI. Deployed the same dataset on vertex AI and trained the model where the results are accuracy is 61% while precision is 75% and Recall 35%. In this scenario Our classifier is very picky, and does not think many things are “happy”,  for example  All the images it thinks are “happy”, are really “happy”. However it also misses a lot of actual “happy”, because it is so very picky. We have low recall, but a very high precision.
The accuracy of happy is 95% surprise 85% neutral 65%, sad 62%
Our classifier is very picky, and does not think many things are hot dogs. All the images it thinks are “happy”, are really “happy”. However it also misses a lot of actual “happy”, because it is so very picky. We have low recall, but a very high precision.

Later I have taken 3 entirely different classes angry, happy, and shock which worked the accuracy is 98% where the  Recall 91% and precision 91%. Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an model returns most of the relevant results. So I have used the model endpoint on flask and prediction was correct. we used the same logic to our previous model. which would be explained in experiment 3 

***Experiment 3:***
There are about 3 classes namely `angry','neutral' and 'sad'. 
* A Sequential model was created.
* Convolutional Neural Networks Conv2D layers are  added inside the sequential model with filters , padding , activation function(`RELU`) and the input shape.
* `MaxPooling` is used to extract the best features from the output of Conv2D which helps in reducing Overfitting.
* `Dropout` is used to manage and balance the accuracy between train and validation sets by increasing or decreasing probability.
* Flatten is used to convert the output of before layer to 1-D Array.
* 2 Fully Connected Layers are used and One output layer with activation layer as `softmax`.
* `ModelCheckPoint` is used to save the model at the end of every Epoch and monitor the performance according to mentioned variable
* `EarlyStopping` is used to stop the job to run when there is no improvement in the parameter pass to EarlyStopping Callback.
* `ReduceLOROnPlateau` helps in monitoring the learning rate (maximizes or minimizes or no improvement).


## Conclusion Experiment 3

#### Training dataset Experiment 3

`Epoch 12: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
87/87 [==============================] - 214s 2s/step - loss: 0.6609 - accuracy: 0.7196 - val_loss: 0.8897 - val_accuracy: 0.6210 - lr: 5.0000e-04
Epoch 12: early stopping`

`Epoch 8: ReduceLROnPlateau reducing learning rate to 1.9999999494757505e-06.
87/87 [==============================] - 214s 2s/step - loss: 0.6584 - accuracy: 0.7188 - val_loss: 0.8444 - val_accuracy: 0.6365 - lr: 1.0000e-05
Epoch 8: early stopping`

#### Test dataset Experiment 3
`26/26 [==============================] - 11s 423ms/step - loss: 0.8637 - accuracy: 0.6087
test_loss: 0.863679, test_accuracy: 0.608748`

* Training and Testing dataset accuracy was very less while using normal accuracy but with CNN model accuracy randomly increased by managing layers and experimenting with optimizers, learning rate , dropout probablity.

### Training Experiment 3
* By using `.fit` , trained the training set with the sequential model and mentioned validation set during for `validation_accuracy` and `validation_loss`.
* Maximum Accuracy of the training and validation is 71%.
* Mentioned the epochs and the batch size.
* By using `.evaluate` , tested the accuracy and loss of testing data with sequential model.
* Accuracy was very less initially due to Image MultiClassification, with the help of new convulutional layers and Dropout Functions , Accuracy has been increased.
* By Comparing the validation and Training loss, experimented with different learning rates through which test , validation and test accuracy constantly increased.

## Testing Experiment3
*`test_dir` is a test dataset which consists of 3 classes `angry, neutral , sad` .
* `ImageDataGenerator` is created by using which random transformations have been performed with the help of `flow_from_directory`.
* By using the model through which data is trained `model` used to predict the test dataset and the maximum accuracy witnessed was 64%.
* Generated a Confusion Matrix by using `model.predict(test_data)` and data visualization was provided for Confusion Matrix.
* Implemented a function named `predict_label` and predict single image classification which helped in understading the model and data efficiency.
* MisClassification of image also enccountered while predicting few images.

**Run Flask app on local system:**
Flask app for local development is using the flask run command from a terminal. By default, Flask will run the application you defined in app.py on port 5000. While the application is running, go to http://localhost:5000 using your web browser. As we are in the development stage, To reload your application automatically whenever you make a change to it. You can do this by passing an environment variable, FLASK_ENV=development, to flask run.
* Saved the model used for training into `face_emotion_e.h5`.
* Created a `app.py` and installed `flask` using `pip. Create a virtual environment and activate for running the flask app.
* For Creating a web page GET and POST methods are used to upload images and display the prediction .
* Images uploaded are saved into the `static` folder and a folder named templates is created which consists of HTML file used.
* In main mention `app.run`. Run the Flask using `flask run` in the virtual environment.

**Heroku Deployment**
* Login To heroku account, install heroku CLI.
* create a file named Procfile in the project’s root directory. This file tells Heroku how to run the app.
* install Gunicorn and update the requirements.txt file using pip
heroku create face-emo-apps command initializes the Heroku application
* git push heroku master pushing the master branch to the heroku remote
* heroku open will open your application using your default web browser. 

## Improvments to be done
* Accuracy of the Image Classification can be increased by balancing the classes with class weights , data augmentation.
* Very Deep learning Models like VGG16 , ResiNet would increase the accuracy and F1-Score of the data.
* Model can be tested using OPENCV which helps in predicting the Real Time Face Expressions by using WebCam.

**Reference**
https://www.analyticsvidhya.com/blog/2021/11/facial-emotion-detection-using-cnn/
https://realpython.com/flask-by-example-part-1-project-setup/
https://github.com/karen-jin/vehicle-classification-TF
https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f
https://www.youtube.com/watch?v=Bb4Wvl57LIk

