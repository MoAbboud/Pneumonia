# Pneumonia Detector
Python Project 2022

Video demo: https://youtu.be/vZO1iLW8xDw

Our project was created for the python programming course given at UMKC. This project has been submitted to the Spring 2022 UDICA Hackathon.

We created a Convolotional Neural Network model and trained it using the Chest X-Ray Images (Pneumonia):

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 

The dataset was already split between train and test data, all we had to do was to preprocess and normalize the images and then fit them on our
created Convolutional Neural Network model.

For this project we built a Pneumonia Detector web application using Flask. Our model "PNmodel.h5" was saved and stored inside the web app folder called PneumoniaDetectorWebApp. This web app will represent the user interface for the target audience who are doctors and medical professionals, that will need
assistance in determining if patients suffer from pnuemonia or not.

The user interface consists of 3 tabs. The first tab is for the user to upload the image by prompting them the file upload directory, after choosing the
image, the user will then click submit. The second tab on the right is a container to display the image after the user clicks submit. The image displayed
is the original size to allow the user to review the image clearly. The third tab is to display the result of the prediction whether the model was able
to predict whether the chest X-ray of the patient has pneumonia or not.

To run this project, download the PneumoniaDetectorWebApp folder that contains the flask app. Open the project using a Pycharm or any IDE that supports 
Python. Run the main.py folder and use http://127.0.0.1:5000/ in a new tab in the browser to display the user interface. The test Chest X-rays can be found
in the x-rays folder. 



