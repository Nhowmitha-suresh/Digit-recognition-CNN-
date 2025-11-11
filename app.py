from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2, base64, os

# ==================================================
# Flask App Initialization
# ==================================================
app = Flask(__name__)

# ==================================================
# Model Loading with Auto-Detection (.h5 or .keras)
# ==================================================
model_path_h5 = 'model/digit_model.h5'
model_path_keras = 'model/digit_model.keras'

if os.path.exists(model_path_h5):
    print("✅ Loading model from:", model_path_h5)
    model = load_model(model_path_h5)
elif os.path.exists(model_path_keras):
    print("✅ Loading model from:", model_path_keras)
    model = load_model(model_path_keras)
else:
    raise FileNotFoundError("❌ Model file not found! Please train and save your model first.")

# ==================================================
# Home Route
# ==================================================
@app.route('/')
def home():
    return render_template('index.html')

# ==================================================
# Prediction Route
# ==================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.get_json().get('image', None)
        if not data:
            return jsonify({'error': 'No image data received'}), 400

        # Decode base64 image
        image_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        # Validate image shape
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Preprocess for model
        img = cv2.resize(img, (28, 28))
        img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0

        # Predict
        probs = model.predict(img)[0]
        prediction = int(np.argmax(probs))

        # Return as JSON
        return jsonify({
            'prediction': prediction,
            'probabilities': [float(p) for p in probs]
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({'error': str(e)}), 500

# ==================================================
# Run Flask Server
# ==================================================
if __name__ == "__main__":
    print("🚀 Starting Flask server... Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True)
