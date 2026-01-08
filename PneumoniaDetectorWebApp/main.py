import numpy as np
import os
import logging
import gc
from flask import Flask, render_template, request, flash

# TensorFlow memory optimization
import tensorflow as tf
from keras import backend as K
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import load_model
from keras.preprocessing import image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # For flash messages

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Model paths
MODEL_PATHS = {
    'Custom CNN': 'PNmodel.h5',
    'ResNet50': 'PNmodelResNet50.h5',
    'VGG16': 'PNmodelVgg.h5'
}

# Model cache - load on demand to save memory
_model_cache = {}

def clear_model_cache():
    """Aggressively clear all models from memory"""
    global _model_cache
    logger.info("Clearing model cache...")
    for model_name in list(_model_cache.keys()):
        del _model_cache[model_name]
    _model_cache = {}
    K.clear_session()
    gc.collect()
    logger.info("Model cache cleared!")

def get_model(model_name):
    """Load model on demand (lazy loading to save memory)"""
    if model_name not in _model_cache:
        # Clear old models before loading new one
        clear_model_cache()
        logger.info(f"Loading {model_name} model...")
        _model_cache[model_name] = load_model(MODEL_PATHS[model_name])
        logger.info(f"{model_name} model loaded!")
    return _model_cache[model_name]

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

def get_all_predictions(img_path, selected_model_only=False, model_name='Custom CNN'):
    """Get predictions - optimized for low memory"""
    predictions = {}
    
    try:
        logger.info(f"Starting predictions for image: {img_path}")
        
        # Load image once
        x = image.load_img(img_path, target_size=(150, 150))
        x_array = image.img_to_array(x) / 255
        
        if selected_model_only:
            # Only predict with selected model to save memory
            logger.info(f"Making {model_name} prediction only...")
            model_obj = get_model(model_name)
            
            if model_name == 'Custom CNN':
                x_input = x_array.reshape(-1, 150, 150, 1)
                pred = model_obj.predict(x_input, verbose=0)[0, 0]
                threshold = THRESHOLDS['CM']
            else:
                x_input = np.expand_dims(x_array, axis=0)
                pred = model_obj.predict(x_input, verbose=0)[0, 0]
                threshold = THRESHOLDS['RM'] if model_name == 'ResNet50' else THRESHOLDS['VM']
            
            predictions[model_name] = {
                'confidence': float(pred * 100),
                'result': result[(pred > threshold).astype("int32")],
                'threshold': threshold
            }
        else:
            # Predict with all models (memory intensive!)
            # Custom CNN Model
            logger.info("Making Custom CNN prediction...")
            x_reshaped = x_array.reshape(-1, 150, 150, 1)
            cm_pred = get_model('Custom CNN').predict(x_reshaped, verbose=0)[0, 0]
            predictions['Custom CNN'] = {
                'confidence': float(cm_pred * 100),
                'result': result[(cm_pred > THRESHOLDS['CM']).astype("int32")],
                'threshold': THRESHOLDS['CM']
            }
            logger.info(f"Custom CNN result: {predictions['Custom CNN']}")
            
            # ResNet50 Model
            logger.info("Making ResNet50 prediction...")
            x_expanded = np.expand_dims(x_array, axis=0)
            rn_pred = get_model('ResNet50').predict(x_expanded, verbose=0)[0, 0]
            predictions['ResNet50'] = {
                'confidence': float(rn_pred * 100),
                'result': result[(rn_pred > THRESHOLDS['RM']).astype("int32")],
                'threshold': THRESHOLDS['RM']
            }
            logger.info(f"ResNet50 result: {predictions['ResNet50']}")
            
            # VGG16 Model
            logger.info("Making VGG16 prediction...")
            vgg_pred = get_model('VGG16').predict(x_expanded, verbose=0)[0, 0]
            predictions['VGG16'] = {
                'confidence': float(vgg_pred * 100),
                'result': result[(vgg_pred > THRESHOLDS['VM']).astype("int32")],
                'threshold': THRESHOLDS['VM']
            }
            logger.info(f"VGG16 result: {predictions['VGG16']}")
        
        logger.info("Predictions completed successfully")
        
        # Aggressive memory cleanup after prediction
        del x, x_array
        if 'x_reshaped' in locals():
            del x_reshaped
        if 'x_expanded' in locals():
            del x_expanded
        gc.collect()
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in get_all_predictions: {str(e)}", exc_info=True)
        raise


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('Home.html')


@app.route("/submit", methods=['POST'])
def performPrediction():
    try:
        logger.info("Received prediction request")
        
        if request.method == 'POST':
            # Check if file was uploaded
            if 'my_image' not in request.files:
                logger.warning("No file in request")
                flash('No file uploaded', 'error')
                return render_template("Home.html")
            
            xray_img = request.files['my_image']
            logger.info(f"File received: {xray_img.filename}")
            
            # Check if filename is empty
            if xray_img.filename == "":
                logger.warning("Empty filename")
                flash('No file selected', 'error')
                return render_template("Home.html")
            
            # Validate file type
            if not allowed_file(xray_img.filename):
                logger.warning(f"Invalid file type: {xray_img.filename}")
                flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)', 'error')
                return render_template("Home.html")
            
            # Save the uploaded file
            xray_img_path = "static/" + xray_img.filename
            logger.info(f"Saving file to: {xray_img_path}")
            xray_img.save(xray_img_path)
            
            # Get selected model
            modelSelection = request.form.get('options', 'CM')
            model_names = {'CM': 'Custom CNN', 'RM': 'ResNet50', 'VM': 'VGG16'}
            selected_model_name = model_names.get(modelSelection, 'Custom CNN')
            logger.info(f"Selected model: {selected_model_name}")
            
            # MEMORY OPTIMIZATION: Only predict with selected model to avoid OOM
            logger.info(f"Getting prediction from {selected_model_name} only...")
            all_predictions = get_all_predictions(xray_img_path, selected_model_only=True, model_name=selected_model_name)
            
            # Get primary prediction from selected model
            primary_prediction = all_predictions[selected_model_name]
            
            logger.info("Rendering results...")
            return render_template("Home.html", 
                                 xray_prediction=primary_prediction['result'],
                                 confidence=primary_prediction['confidence'],
                                 xray_img_path=xray_img_path,
                                 all_predictions=all_predictions,
                                 selected_model=selected_model_name)
        
        return render_template("Home.html")
        
    except Exception as e:
        logger.error(f"Error in performPrediction: {str(e)}", exc_info=True)
        flash(f'Error processing image: {str(e)}', 'error')
        return render_template("Home.html"), 500


if __name__ == "__main__":
    app.run(debug=True)
