from flask import Flask, request, jsonify, render_template, make_response
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.layers import Layer


main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')  
models_dir = os.path.join(main_dir, 'models')
uploads_dir = os.path.join(main_dir, 'webapp', 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = uploads_dir

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Path to the models folder (one level up from webapp)
MODELS_FOLDER = models_dir

# Path to the class indices file (one level up from webapp)
CLASS_INDICES_FILE = os.path.join(main_dir, 'class_indices.json')

def load_class_indices():
    """Load the class indices from the JSON file."""
    with open(CLASS_INDICES_FILE, 'r') as f:
        class_indices = json.load(f)
    # Invert the dictionary to map indices to class labels
    index_to_class = {v: k for k, v in class_indices.items()}
    return index_to_class

def get_model_files():
    """Get a list of all .h5 files in the models folder without the .h5 extension."""
    model_files = {}
    for file_name in os.listdir(MODELS_FOLDER):
        if file_name.endswith(".h5"):
            model_name = file_name[:-3]  # Remove the .h5 extension
            model_files[model_name] = os.path.join(MODELS_FOLDER, file_name)
    return model_files

def load_selected_model(model_name):
    model_files = get_model_files()
    if model_name in model_files:
        return load_model(model_files[model_name])
    else:
        return None

def prepare_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    model_files = get_model_files()
    last_model = request.cookies.get('selected_model', next(iter(model_files)))

    if request.method == "POST":
        model_name = request.form.get("model", next(iter(model_files)))
        file = request.files.get("file")

        if file and model_name:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            model = load_selected_model(model_name)
            if not model:
                return jsonify({"error": "Model not found"}), 400

            image = prepare_image(file_path)
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]

            # Load the class indices
            index_to_class = load_class_indices()

            # Map the predicted index to the breed name
            breed_name = index_to_class.get(predicted_class, "Unknown")

            response = make_response(jsonify({"breed": breed_name}))
            response.set_cookie('selected_model', model_name)

            return response

    return render_template("index.html", models=model_files.keys(), last_model=last_model)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
