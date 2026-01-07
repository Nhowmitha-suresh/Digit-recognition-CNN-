import base64, cv2, numpy as np, os
from flask import Flask, render_template, request, jsonify
from scipy import ndimage
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("model/digit_model.h5")

def get_best_shift(img):
    """Aligns the digit's center of mass to the center of the image."""
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    return np.round(cols/2.0 - cx).astype(int), np.round(rows/2.0 - cy).astype(int)

def preprocess_digit(img):
    # 1. Thicken line and Threshold
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Extract digit bounding box
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y+h, x:x+w]

    # 3. Resize to 20x20 while keeping aspect ratio
    f = 20 / max(w, h)
    digit = cv2.resize(digit, (int(w*f), int(h*f)), interpolation=cv2.INTER_AREA)

    # 4. Center into 28x28 using Center of Mass
    nh, nw = digit.shape
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[(28-nh)//2:(28-nh)//2+nh, (28-nw)//2:(28-nw)//2+nw] = digit
    sx, sy = get_best_shift(padded)
    
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    padded = cv2.warpAffine(padded, M, (28, 28))
    return padded.astype("float32").reshape(1, 28, 28, 1) / 255.0

@app.route("/")
def home(): return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()["image"].split(",")[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_not(img) # Convert white-bg to black-bg for MNIST
        
        processed = preprocess_digit(img)
        if processed is None: return jsonify({"digit": "-", "confidence": 0})
        
        preds = model.predict(processed, verbose=0)[0]
        return jsonify({"digit": int(np.argmax(preds)), "confidence": round(float(np.max(preds))*100, 2)})
    except Exception as e: return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)