import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy import ndimage
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("model/digit_model.h5")

def get_best_shift(img):
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0 - cx).astype(int)
    shifty = np.round(rows/2.0 - cy).astype(int)
    return shiftx, shifty

def preprocess_digit(img):
    # 1. Image Enhancement
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    
    # 2. Advanced Segmentation
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Contour Detection & Cropping
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y+h, x:x+w]

    # 4. Standardized Resizing (MNIST Style)
    f = 20 / max(w, h)
    digit = cv2.resize(digit, (int(w*f), int(h*f)), interpolation=cv2.INTER_AREA)

    # 5. Mass-Based Centering
    padded = np.zeros((28, 28), dtype=np.uint8)
    nh, nw = digit.shape
    padded[(28-nh)//2:(28-nh)//2+nh, (28-nw)//2:(28-nw)//2+nw] = digit
    sx, sy = get_best_shift(padded)
    
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    final = cv2.warpAffine(padded, M, (28, 28))
    return final.astype("float32").reshape(1, 28, 28, 1) / 255.0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_data = request.get_json()["image"].split(",")[1]
        img_arr = np.frombuffer(base64.b64decode(raw_data), np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
        
        # Live canvas is usually White-BG; MNIST needs Black-BG
        img = cv2.bitwise_not(img) 
        
        processed = preprocess_digit(img)
        if processed is None: return jsonify({"digit": "Empty", "conf": 0})
        
        res = model.predict(processed, verbose=0)[0]
        return jsonify({
            "digit": int(np.argmax(res)),
            "conf": round(float(np.max(res)) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=5000, debug=True)