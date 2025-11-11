# Digit Recognition using CNN (Flask Web App)

A deep learning web application that recognizes hand-drawn digits (0–9) in real time using a **Convolutional Neural Network (CNN)** built with TensorFlow and deployed using Flask.

---

## 📘 Overview

This project demonstrates the use of deep learning for digit classification through an interactive web interface.  
The CNN model was trained on the **MNIST dataset** and achieves around **98.85% accuracy**.  
Users can draw digits on a virtual canvas and get instant predictions.

---

## 🧠 Features

- Convolutional Neural Network (CNN) trained on MNIST  
- Flask backend for real-time prediction  
- Interactive digit drawing canvas using JavaScript  
- Clear and Predict buttons for testing  
- Responsive, minimal user interface  

---

## 🧩 Technologies Used

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Programming Language** | Python |
| **Deep Learning** | TensorFlow / Keras |
| **Web Framework** | Flask |
| **Frontend** | HTML, CSS, JavaScript (Canvas API) |
| **Dataset** | MNIST Handwritten Digits |

---
## ⚙️ Installation & Setup

Follow the steps below to run this project locally.

---

### 🧩 Step 1: Clone the Repository
Clone the repository from GitHub using the following command:
git clone https://github.com/Nhowmitha-suresh/Digit-recognition-CNN-.git

Then, navigate into the project directory:
cd "Digit Recognition (CNN)"

---

### 🧩 Step 2: Create a Virtual Environment
Create a virtual environment to isolate project dependencies:
python -m venv .venv

---

### 🧩 Step 3: Activate the Virtual Environment
Activate the virtual environment before installing dependencies.

On Windows:
.venv\Scripts\activate

On macOS / Linux:
source .venv/bin/activate

---

### 🧩 Step 4: Install Dependencies
Install all required Python packages using the requirements.txt file:
pip install -r requirements.txt

This will install:
- TensorFlow  
- Flask  
- NumPy  
- OpenCV  
- Matplotlib  
- and other dependencies.

---

### 🧩 Step 5: Run the Flask Application
Start the Flask web server with the following command:
python app.py

Once the server starts, you’ll see something like this:
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

---

### 🧩 Step 6: Open the Application in Browser
Open your browser and visit:
http://127.0.0.1:5000

You’ll see the Digit Recognition web interface.
Draw a digit (0–9) and click Predict to view the model’s prediction.
Use Clear to reset the canvas and try again.

---

✅ Your setup is complete!
The application is now running successfully on your local machine.
