import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy import ndimage
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model from your directory structure
MODEL_PATH = "model/digit_model.h5"
try:
    model = load_model(MODEL_PATH)
    print("✅ High-accuracy model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def get_best_shift(img):
    """Calculates the center of mass to align digit with MNIST training style."""
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy):
    """Shifts the image based on center of mass offsets."""
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(img, M, (cols, rows))

def preprocess_digit(img):
    """Full MNIST-standard preprocessing pipeline."""
    # 1. Thicken line to help the model 'see' thin canvas strokes
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # 2. Thresholding (Ensuring white digit on black background)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 3. Find digit and crop to bounding box
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y+h, x:x+w]

    # 4. Resize to 20x20 while maintaining aspect ratio
    if w > h:
        new_w, new_h = 20, int(h * (20 / w))
    else:
        new_h, new_w = 20, int(w * (20 / h))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. Pad into 28x28 frame
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[(28 - new_h) // 2 : (28 - new_h) // 2 + new_h, 
           (28 - new_w) // 2 : (28 - new_w) // 2 + new_w] = digit

    # 6. Apply Center of Mass shift (The accuracy secret)
    sx, sy = get_best_shift(padded)
    padded = shift(padded, sx, sy)

    # 7. Normalize pixel values [0, 1]
    return padded.astype("float32").reshape(1, 28, 28, 1) / 255.0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"digit": "", "confidence": 0})

        # Decode base64 image from canvas
        image_data = data["image"].split(",")[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Invert colors if canvas is white-background (MNIST is white-on-black)
        img = cv2.bitwise_not(img)

        processed = preprocess_digit(img)
        if processed is None:
            return jsonify({"digit": "Empty", "confidence": 0})

        # Get prediction and probability
        preds = model.predict(processed, verbose=0)[0]
        digit = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        return jsonify({
            "digit": digit,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"digit": "Error", "confidence": 0})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)