# flask_app.py
import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model  # type: ignore

app = Flask(__name__)

# Load the saved model
model = load_model('pneumonia_model_final.keras')

def preprocess_image(image):
    """Preprocess the image for prediction."""
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    if img is None:
        return None
    
    img = cv2.resize(img, (224, 224))  # Resize to 224x224 pixels
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1] range
    img = (img * 255).astype(np.uint8)  # Convert back to 8-bit unsigned integer format
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def home():
    """Render the home page for uploading images."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    
    # Save the uploaded file temporarily
    file_path = 'temp_image.jpg'
    file.save(file_path)

    # Preprocess the image
    processed_image = preprocess_image(file_path)
    if processed_image is None:
        return jsonify({'error': 'Invalid image format.'}), 400

    # Make a prediction
    prediction = model.predict(processed_image)
    probability = prediction[0][0]  # Get the probability score

    # Calculate percentage probability
    probability_percent = round(probability * 100, 2)
    
    # Classification threshold
    threshold = 0.7  # Adjust threshold if necessary
    is_pneumonia = probability > threshold

    # Determine result message
    result_text = f"Pneumonia" if is_pneumonia else f"Not Pneumonia"

    # Remove the temporary file
    os.remove(file_path)

    # Render the result page with prediction details
    return render_template('result.html', result=result_text, probability=probability_percent)

if __name__ == '__main__':
    app.run(debug=True)
