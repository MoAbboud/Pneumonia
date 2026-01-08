import numpy as np
import os
import logging
from flask import Flask, render_template, request, flash
from keras.models import load_model
from keras.preprocessing import image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # For flash messages

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Store models as None initially - load them lazily
models_cache = {
    'CM': None,
    'RM': None,
    'VM': None
}

result = {0: 'Pneumonia', 1: 'Normal'}

# Model thresholds
THRESHOLDS = {
    'CM': 0.25,
    'RM': 0.313,
    'VM': 0.3
}

def load_model_lazy(model_type):
    """Load model only when needed to save memory"""
    if models_cache[model_type] is None:
        if model_type == 'CM':
            models_cache['CM'] = load_model('PNmodel.h5')
        elif model_type == 'RM':
            models_cache['RM'] = load_model('PNmodelResNet50.h5')
        elif model_type == 'VM':
            models_cache['VM'] = load_model('PNmodelVgg.h5')
    return models_cache[model_type]

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_all_predictions(img_path):
    """Get predictions from all three models"""
    import gc
    predictions = {}
    
    try:
        logger.info(f"Starting predictions for image: {img_path}")
        
        # Custom CNN Model
        logger.info("Loading Custom CNN model...")
        model = load_model_lazy('CM')
        x = image.load_img(img_path, target_size=(150, 150))
        x_array = image.img_to_array(x) / 255
        x_reshaped = x_array.reshape(-1, 150, 150, 1)
        logger.info("Making Custom CNN prediction...")
        cm_pred = model.predict(x_reshaped, verbose=0)[0, 0]
        predictions['Custom CNN'] = {
            'confidence': float(cm_pred * 100),
            'result': result[(cm_pred > THRESHOLDS['CM']).astype("int32")],
            'threshold': THRESHOLDS['CM']
        }
        logger.info(f"Custom CNN result: {predictions['Custom CNN']}")
        del model
        models_cache['CM'] = None
        gc.collect()
        
        # ResNet50 Model
        logger.info("Loading ResNet50 model...")
        modelResNet50 = load_model_lazy('RM')
        x = image.load_img(img_path, target_size=(150, 150))
        x_array = image.img_to_array(x) / 255
        x_expanded = np.expand_dims(x_array, axis=0)
        logger.info("Making ResNet50 prediction...")
        rn_pred = modelResNet50.predict(x_expanded, verbose=0)[0, 0]
        predictions['ResNet50'] = {
            'confidence': float(rn_pred * 100),
            'result': result[(rn_pred > THRESHOLDS['RM']).astype("int32")],
            'threshold': THRESHOLDS['RM']
        }
        logger.info(f"ResNet50 result: {predictions['ResNet50']}")
        del modelResNet50
        models_cache['RM'] = None
        gc.collect()
        
        # VGG16 Model
        logger.info("Loading VGG16 model...")
        modelVgg19 = load_model_lazy('VM')
        logger.info("Making VGG16 prediction...")
        vgg_pred = modelVgg19.predict(x_expanded, verbose=0)[0, 0]
        predictions['VGG16'] = {
            'confidence': float(vgg_pred * 100),
            'result': result[(vgg_pred > THRESHOLDS['VM']).astype("int32")],
            'threshold': THRESHOLDS['VM']
        }
        logger.info(f"VGG16 result: {predictions['VGG16']}")
        del modelVgg19
        models_cache['VM'] = None
        gc.collect()
        
        logger.info("All predictions completed successfully")
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
            
            # Get predictions from all models
            logger.info("Getting predictions from all models...")
            all_predictions = get_all_predictions(xray_img_path)
            
            # Get selected model
            modelSelection = request.form.get('options', 'CM')
            logger.info(f"Selected model: {modelSelection}")
            
            # Map selection to model name
            model_names = {'CM': 'Custom CNN', 'RM': 'ResNet50', 'VM': 'VGG16'}
            selected_model_name = model_names.get(modelSelection, 'Custom CNN')
            
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
