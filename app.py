from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the trained model
model = load_model('pet_disease_cnn_rnn_model.h5')

# Class names or labels
class_names = ['Bacterial_dermatosis', 'Fungal_infections', 'Healthy', 'Hypersensitivity_allergic_dermatosis']  # Replace with actual class names

app = Flask(__name__)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Route for receiving image and processing it
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_label = class_names[np.argmax(prediction)]
            return render_template('result.html', label=predicted_label)
    return render_template('index.html')

# Route for the result page
@app.route('/result')
def result():
    return render_template('result.html')


