import os
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('pet_disease_cnn_rnn_model.h5')

# Define the label names based on the trained model's classes
LABELS = ['Bacterial_dermatosis', 'Fungal_infections', 'Healthy', 'Hypersensitivity_allergic_dermatosis']  # Replace with actual class names
import os

def prepare_image(file_path):
    """Load and preprocess the image."""
    image = load_img(file_path, target_size=(224, 224))  # Resize to model input size
    image = img_to_array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file and file.filename != '':
        # Save the uploaded image to a temporary location
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Prepare the image for prediction
        image = prepare_image(file_path)

        # Predict the label of the image
        predictions = model.predict(image)
        predicted_label_index = np.argmax(predictions)
        predicted_label = LABELS[predicted_label_index]

        # Remove the temporary image
        os.remove(file_path)

        return render_template('index.html', label=predicted_label)

    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
