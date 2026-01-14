# Pneumonia Detector
Python Project 2022

Video demo: https://youtu.be/OHRrb3LttRY

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

## New Features

### 🤖 Multi-Model AI Analysis
- **Three AI Models**: Added ResNet50 and VGG16 alongside the original Custom CNN
- **Ensemble Prediction**: Weighted ensemble combining all three models for improved accuracy
- **Model Performance Metrics**: Displays accuracy, precision, recall, and F1-score for each model
- **Side-by-Side Comparison**: View predictions from all models simultaneously

### 📊 Enhanced Results Dashboard
- **Confidence Gauges**: Circular progress indicators showing each model's confidence level
- **Model Agreement Indicators**: Shows consensus between models (e.g., "All 3 models agree")
- **Color-Coded Results**: Green for Normal, red for Pneumonia
- **Detailed Metrics**: Performance statistics for each model displayed with predictions

### 🎨 Modern User Interface
- **Dark/Light Mode Toggle**: Switch between themes with persistent preference storage
- **Animated Sample Image Sliders**: Infinite scrolling galleries displaying 20+ sample X-rays
- **Drag & Drop Upload**: Modern file upload interface with instant image preview
- **Responsive Design**: Optimized for both desktop and mobile devices
- **Loading Animations**: Professional loading overlay during AI analysis

### 🖼️ Interactive Sample Images
- **Click-to-Load**: Click any sample X-ray image to instantly load it for analysis
- **20+ Pre-loaded Samples**: Mix of Normal (10) and Pneumonia (10) cases
- **Visual Labels**: Each sample clearly labeled as Normal or Pneumonia
- **Dual Sliders**: Sample images displayed on both left and right sides of the upload area

### ✅ Smart Image Validation
- **Resolution Check**: Validates minimum image size (150x150 pixels)
- **Quality Assessment**: Detects corrupted, blank, or low-quality images
- **Multiple Format Support**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **File Size Validation**: Ensures images aren't corrupted based on file size

### 🎯 Improved User Experience
- **Medical Disclaimer Banner**: Prominent warning that tool is for educational purposes only
- **Flash Message System**: Clear feedback for errors and successful operations
- **Selected Sample Indicator**: Shows which sample image is currently selected with option to clear
- **File Upload Hints**: Helpful text guiding users on supported formats and sizes 



