import os
import numpy as np
import ast
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
IMG_SIZE = 224

# Load class indices
with open("class_indices.txt", "r") as f:
    class_indices = ast.literal_eval(f.read())
class_labels = {v: k for k, v in class_indices.items()}

# Lazy load models to save memory initially
models = {
    "MobileNetV2": None,
    "ResNet50": None
}

def get_model(name):
    if models[name] is None:
        if name == "MobileNetV2":
            models[name] = load_model("models/currency_model.h5")
        elif name == "ResNet50":
            models[name] = load_model("models/currency_resnet_model.h5")
    return models[name]

def preprocess_image(img_path, model_name):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_name == "MobileNetV2":
        img_array /= 255.0
    elif model_name == "ResNet50":
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    model_name = request.form.get('model', 'MobileNetV2')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        model = get_model(model_name)
        img_array = preprocess_image(filepath, model_name)
        
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        result = {
            "prediction": class_labels[class_idx],
            "confidence": f"{confidence * 100:.2f}%",
            "model_used": model_name
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize models on startup (optional, but good for UX)
    print("Loading models...")
    get_model("MobileNetV2")
    # get_model("ResNet50") # ResNet is heavy, maybe load on demand
    app.run(debug=True, port=5000)
