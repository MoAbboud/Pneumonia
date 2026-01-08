import numpy as np
import os
from flask import Flask, render_template, request, flash
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # For flash messages

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

model = load_model('PNmodel.h5')
modelResNet50 = load_model('PNmodelResNet50.h5')
modelVgg19 = load_model('PNmodelVgg.h5')

result = {0: 'Pneumonia', 1: 'Normal'}

# Model thresholds
THRESHOLDS = {
    'CM': 0.25,
    'RM': 0.313,
    'VM': 0.3
}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_all_predictions(img_path):
    """Get predictions from all three models"""
    predictions = {}
    
    # Custom CNN Model
    x = image.load_img(img_path, target_size=(150, 150))
    x_array = image.img_to_array(x) / 255
    x_reshaped = x_array.reshape(-1, 150, 150, 1)
    cm_pred = model.predict(x_reshaped, verbose=0)[0, 0]
    predictions['Custom CNN'] = {
        'confidence': float(cm_pred * 100),
        'result': result[(cm_pred > THRESHOLDS['CM']).astype("int32")],
        'threshold': THRESHOLDS['CM']
    }
    
    # ResNet50 Model
    x = image.load_img(img_path, target_size=(150, 150))
    x_array = image.img_to_array(x) / 255
    x_expanded = np.expand_dims(x_array, axis=0)
    rn_pred = modelResNet50.predict(x_expanded, verbose=0)[0, 0]
    predictions['ResNet50'] = {
        'confidence': float(rn_pred * 100),
        'result': result[(rn_pred > THRESHOLDS['RM']).astype("int32")],
        'threshold': THRESHOLDS['RM']
    }
    
    # VGG16 Model
    vgg_pred = modelVgg19.predict(x_expanded, verbose=0)[0, 0]
    predictions['VGG16'] = {
        'confidence': float(vgg_pred * 100),
        'result': result[(vgg_pred > THRESHOLDS['VM']).astype("int32")],
        'threshold': THRESHOLDS['VM']
    }
    
    return predictions


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('Home.html')


@app.route("/submit", methods=['POST'])
def performPrediction():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'my_image' not in request.files:
            flash('No file uploaded', 'error')
            return render_template("Home.html")
        
        xray_img = request.files['my_image']
        
        # Check if filename is empty
        if xray_img.filename == "":
            flash('No file selected', 'error')
            return render_template("Home.html")
        
        # Validate file type
        if not allowed_file(xray_img.filename):
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)', 'error')
            return render_template("Home.html")
        
        try:
            # Save the uploaded file
            xray_img_path = "static/" + xray_img.filename
            xray_img.save(xray_img_path)
            
            # Get predictions from all models
            all_predictions = get_all_predictions(xray_img_path)
            
            # Get selected model
            modelSelection = request.form.get('options', 'CM')
            
            # Map selection to model name
            model_names = {'CM': 'Custom CNN', 'RM': 'ResNet50', 'VM': 'VGG16'}
            selected_model_name = model_names.get(modelSelection, 'Custom CNN')
            
            # Get primary prediction from selected model
            primary_prediction = all_predictions[selected_model_name]
            
            return render_template("Home.html", 
                                 xray_prediction=primary_prediction['result'],
                                 confidence=primary_prediction['confidence'],
                                 xray_img_path=xray_img_path,
                                 all_predictions=all_predictions,
                                 selected_model=selected_model_name)
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'error')
            print(f"Error: {str(e)}")
            return render_template("Home.html")

    return render_template("Home.html")


if __name__ == "__main__":
    app.run(debug=True)
