# Emotion Detection Web App 🎭🎙️

This is a simple web application that detects emotions from **audio recordings** using a pre-trained deep learning model. The project combines **Flask**, **TensorFlow**, and **Librosa** to extract features from audio and classify them into different emotions.

---

## 🔍 Features

- 🎤 Upload or record voice audio
- 🤖 Predicts emotion using a trained deep learning model
- 📊 Emotions supported: `angry`, `happy`, `neutral`, `sad`, `calm`, `fearful`, `disgust`, `surprised`
- 🧠 Uses MFCC (Mel Frequency Cepstral Coefficients) features
- 💻 Built with Flask, TensorFlow, Numpy, and Librosa

---

## 📁 Project Structure

1. static/ # Static files
2. 2templates/ # HTML
3. emotion_model.h5 # Trained deep learning model
4. app.py # Flask backend logic

---

## Create a virtual environment and activate it

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

---

## Install dependencies

pip install -r requirements.txt

---

##  Run the app

python app.py
