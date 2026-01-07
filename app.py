from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from scipy import ndimage
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("model/digit_model.h5")

def get_best_shift(img):
    """Calculates the center of mass to center the digit perfectly."""
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0 - cx).astype(int)
    shifty = np.round(rows/2.0 - cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy):
    """Applies the shift to the image."""
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(img, M, (cols, rows))

def preprocess_digit(img):
    # 1. Thicken the drawing (dilatation helps if the user used a thin brush)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # 2. Thresholding and finding the digit contour
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y+h, x:x+w]

    # 3. Resize to 20x20 while maintaining aspect ratio (Standard MNIST)
    if w > h:
        new_w, new_h = 20, int(h * (20 / w))
    else:
        new_h, new_w = 20, int(w * (20 / h))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. Place 20x20 digit inside 28x28 padding
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[(28-new_h)//2 : (28-new_h)//2+new_h, (28-new_w)//2 : (28-new_w)//2+new_w] = digit

    # 5. Final centering using Center of Mass
    sx, sy = get_best_shift(padded)
    padded = shift(padded, sx, sy)

    # 6. Normalize for model input
    return padded.astype("float32").reshape(1, 28, 28, 1) / 255.0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()["image"].split(",")[1]
        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # CANVAS CHECK: If your canvas is white-background/black-ink, 
        # use cv2.bitwise_not to make it white-ink/black-background for MNIST
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_not(img) 

        processed = preprocess_digit(img)
        if processed is None: 
            return jsonify({"digit": "None", "confidence": 0})

        preds = model.predict(processed, verbose=0)[0]
        return jsonify({
            "digit": int(np.argmax(preds)),
            "confidence": round(float(np.max(preds)) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)