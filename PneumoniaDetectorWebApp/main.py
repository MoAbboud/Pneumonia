import numpy as np
import os
import logging
import gc
import cv2
from PIL import Image
from flask import Flask, render_template, request, flash, jsonify

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

# Model performance metrics (from training)
MODEL_METRICS = {
    'Custom CNN': {
        'accuracy': 89.5,
        'precision': 91.2,
        'recall': 87.8,
        'f1_score': 89.5,
        'description': 'Lightweight custom architecture, fast inference'
    },
    'ResNet50': {
        'accuracy': 92.3,
        'precision': 93.1,
        'recall': 91.5,
        'f1_score': 92.3,
        'description': 'Deep residual network, balanced performance'
    },
    'VGG16': {
        'accuracy': 91.8,
        'precision': 92.5,
        'recall': 91.0,
        'f1_score': 91.7,
        'description': 'Very deep network, excellent feature extraction'
    }
}

# Ensemble weights (based on model performance)
ENSEMBLE_WEIGHTS = {
    'Custom CNN': 0.30,
    'ResNet50': 0.38,
    'VGG16': 0.32
}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_quality(img_path):
    """Validate image quality before processing"""
    try:
        # Open image
        img = Image.open(img_path)
        
        # Check if image can be opened
        if img is None:
            return False, "Could not open image file"
        
        # Check minimum resolution
        width, height = img.size
        if width < 150 or height < 150:
            return False, f"Image too small ({width}x{height}). Minimum size: 150x150 pixels"
        
        # Check file size (not too small = likely corrupted)
        file_size = os.path.getsize(img_path)
        if file_size < 5000:  # Less than 5KB
            return False, "Image file too small, possibly corrupted"
        
        # Check if grayscale or RGB (X-rays should be grayscale or can be converted)
        if img.mode not in ['L', 'RGB', 'RGBA']:
            return False, f"Unsupported image mode: {img.mode}"
        
        # Check for blank/solid color images
        img_array = np.array(img.convert('L'))
        if img_array.std() < 5:  # Very low variation = likely blank
            return False, "Image appears to be blank or uniform color"
        
        logger.info(f"Image validation passed: {width}x{height}, {file_size} bytes")
        return True, "Image quality OK"
        
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False, f"Image validation failed: {str(e)}"

def generate_gradcam(img_path, model_name='Custom CNN'):
    """Generate Grad-CAM heatmap for visualization"""
    try:
        logger.info(f"Generating Grad-CAM for {model_name}...")
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255
        
        # Get model
        model_obj = get_model(model_name)
        
        # Prepare input based on model type
        if model_name == 'Custom CNN':
            img_input = img_array.reshape(-1, 150, 150, 1)
        else:
            img_input = np.expand_dims(img_array, axis=0)
        
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model_obj.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            logger.warning("No convolutional layer found for Grad-CAM")
            return None
        
        # Create a model that outputs both the conv layer and final prediction
        grad_model = tf.keras.models.Model(
            inputs=[model_obj.inputs],
            outputs=[last_conv_layer.output, model_obj.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            loss = predictions[0]
        
        # Gradient of the output with respect to conv layer
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        # Weight the channels by gradient importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (150, 150))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Load original image
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (150, 150))
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Save heatmap
        heatmap_filename = f"heatmap_{model_name.replace(' ', '_')}.jpg"
        heatmap_path = os.path.join('static', heatmap_filename)
        cv2.imwrite(heatmap_path, superimposed_img)
        
        logger.info(f"Grad-CAM saved to: {heatmap_path}")
        return heatmap_path
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}", exc_info=True)
        return None

def calculate_model_agreement(predictions):
    """Calculate agreement between models"""
    try:
        # Exclude Ensemble from agreement calculation
        model_results = [pred['result'] for name, pred in predictions.items() if name != 'Ensemble']
        
        if not model_results:
            return None
        
        # Count how many agree
        pneumonia_count = model_results.count('Pneumonia')
        normal_count = model_results.count('Normal')
        total = len(model_results)
        agreement_count = max(pneumonia_count, normal_count)
        
        return {
            'total': total,
            'agreeing': agreement_count,
            'majority_result': 'Pneumonia' if pneumonia_count > normal_count else 'Normal',
            'pneumonia_votes': pneumonia_count,
            'normal_votes': normal_count,
            'unanimous': agreement_count == total
        }
    except Exception as e:
        logger.error(f"Error calculating agreement: {str(e)}")
        return None

def get_all_predictions(img_path, selected_model_only=False, model_name='Custom CNN'):
    """Get predictions - optimized for low memory"""
    predictions = {}
    raw_predictions = {}  # Store raw scores for ensemble
    
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
            
            raw_predictions[model_name] = pred
            predicted_class = result[(pred > threshold).astype("int32")]
            # Calculate confidence as percentage - if Normal (pred > threshold), use pred; if Pneumonia, use (1 - pred)
            if predicted_class == 'Normal':
                confidence = float(pred * 100)
            else:
                confidence = float((1 - pred) * 100)
            
            predictions[model_name] = {
                'confidence': confidence,
                'result': predicted_class,
                'threshold': threshold,
                'raw_score': float(pred)
            }
        else:
            # Predict with all models (memory intensive!)
            # Custom CNN Model
            logger.info("Making Custom CNN prediction...")
            x_reshaped = x_array.reshape(-1, 150, 150, 1)
            cm_pred = get_model('Custom CNN').predict(x_reshaped, verbose=0)[0, 0]
            raw_predictions['Custom CNN'] = cm_pred
            cm_result = result[(cm_pred > THRESHOLDS['CM']).astype("int32")]
            cm_confidence = float(cm_pred * 100) if cm_result == 'Normal' else float((1 - cm_pred) * 100)
            predictions['Custom CNN'] = {
                'confidence': cm_confidence,
                'result': cm_result,
                'threshold': THRESHOLDS['CM'],
                'raw_score': float(cm_pred)
            }
            logger.info(f"Custom CNN result: {predictions['Custom CNN']}")
            
            # ResNet50 Model
            logger.info("Making ResNet50 prediction...")
            x_expanded = np.expand_dims(x_array, axis=0)
            rn_pred = get_model('ResNet50').predict(x_expanded, verbose=0)[0, 0]
            raw_predictions['ResNet50'] = rn_pred
            rn_result = result[(rn_pred > THRESHOLDS['RM']).astype("int32")]
            rn_confidence = float(rn_pred * 100) if rn_result == 'Normal' else float((1 - rn_pred) * 100)
            predictions['ResNet50'] = {
                'confidence': rn_confidence,
                'result': rn_result,
                'threshold': THRESHOLDS['RM'],
                'raw_score': float(rn_pred)
            }
            logger.info(f"ResNet50 result: {predictions['ResNet50']}")
            
            # VGG16 Model
            logger.info("Making VGG16 prediction...")
            vgg_pred = get_model('VGG16').predict(x_expanded, verbose=0)[0, 0]
            raw_predictions['VGG16'] = vgg_pred
            vgg_result = result[(vgg_pred > THRESHOLDS['VM']).astype("int32")]
            vgg_confidence = float(vgg_pred * 100) if vgg_result == 'Normal' else float((1 - vgg_pred) * 100)
            predictions['VGG16'] = {
                'confidence': vgg_confidence,
                'result': vgg_result,
                'threshold': THRESHOLDS['VM'],
                'raw_score': float(vgg_pred)
            }
            logger.info(f"VGG16 result: {predictions['VGG16']}")
            
            # Calculate ensemble prediction
            ensemble_score = sum(raw_predictions[model] * ENSEMBLE_WEIGHTS[model] 
                                for model in raw_predictions.keys())
            ensemble_threshold = 0.3  # Average threshold
            ensemble_result = result[(ensemble_score > ensemble_threshold).astype("int32")]
            ensemble_confidence = float(ensemble_score * 100) if ensemble_result == 'Normal' else float((1 - ensemble_score) * 100)
            predictions['Ensemble'] = {
                'confidence': ensemble_confidence,
                'result': ensemble_result,
                'threshold': ensemble_threshold,
                'raw_score': float(ensemble_score),
                'description': 'Weighted average of all models'
            }
            logger.info(f"Ensemble result: {predictions['Ensemble']}")
        
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
            # Check if a sample image URL was provided
            sample_image_url = request.form.get('sample_image', '')
            
            if sample_image_url:
                # Handle sample image from URL
                logger.info(f"Using sample image: {sample_image_url}")
                
                try:
                    # Check if it's a local static file path
                    if sample_image_url.startswith('/static/'):
                        # Remove leading slash and use the local file path
                        xray_img_path = sample_image_url.lstrip('/')
                        logger.info(f"Using local sample image: {xray_img_path}")
                        
                        # Verify the file exists
                        if not os.path.exists(xray_img_path):
                            raise FileNotFoundError(f"Sample image not found: {xray_img_path}")
                    else:
                        # Handle external URL (download it)
                        import requests
                        from io import BytesIO
                        
                        response = requests.get(sample_image_url, timeout=10)
                        response.raise_for_status()
                        
                        # Save the downloaded image
                        img = Image.open(BytesIO(response.content))
                        xray_img_path = f"static/sample_{hash(sample_image_url)}.jpg"
                        img.save(xray_img_path)
                        logger.info(f"Sample image saved to: {xray_img_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to load sample image: {str(e)}")
                    flash('Failed to load sample image. Please try uploading your own image.', 'error')
                    return render_template("Home.html")
            else:
                # Handle uploaded file
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
            
            # Validate image quality
            is_valid, validation_message = validate_image_quality(xray_img_path)
            if not is_valid:
                logger.warning(f"Image validation failed: {validation_message}")
                flash(f'Image quality issue: {validation_message}', 'error')
                # Clean up uploaded file
                if os.path.exists(xray_img_path):
                    os.remove(xray_img_path)
                return render_template("Home.html")
            
            # Get selected model
            modelSelection = request.form.get('options', 'CM')
            model_names = {'CM': 'Custom CNN', 'RM': 'ResNet50', 'VM': 'VGG16'}
            selected_model_name = model_names.get(modelSelection, 'Custom CNN')
            logger.info(f"Selected model: {selected_model_name}")
            
            # Get predictions from ALL models (we're on $25 plan now!)
            logger.info("Getting predictions from all models...")
            all_predictions = get_all_predictions(xray_img_path, selected_model_only=False)
            
            # Generate Grad-CAM heatmap for selected model
            heatmap_path = generate_gradcam(xray_img_path, selected_model_name)
            
            # Get primary prediction (use Ensemble if all models ran)
            primary_prediction = all_predictions.get('Ensemble', all_predictions[selected_model_name])
            
            # Calculate model agreement
            model_agreement = calculate_model_agreement(all_predictions)
            
            logger.info("Rendering results...")
            return render_template("Home.html", 
                                 xray_prediction=primary_prediction['result'],
                                 confidence=primary_prediction['confidence'],
                                 xray_img_path=xray_img_path,
                                 all_predictions=all_predictions,
                                 selected_model=selected_model_name,
                                 heatmap_path=heatmap_path,
                                 model_metrics=MODEL_METRICS,
                                 model_agreement=model_agreement)
        
        return render_template("Home.html")
        
    except Exception as e:
        logger.error(f"Error in performPrediction: {str(e)}", exc_info=True)
        flash(f'Error processing image: {str(e)}', 'error')
        return render_template("Home.html"), 500


if __name__ == "__main__":
    app.run(debug=True)
