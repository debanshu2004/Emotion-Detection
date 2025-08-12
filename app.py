from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
import librosa
from pydub import AudioSegment
import random
import io
import base64
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Emotion labels (update if your model uses different order)
emotion_labels = ['angry', 'happy', 'neutral', 'sad', 'calm', 'fearful', 'disgust', 'surprised']

# Load the trained model once at startup
MODEL_PATH = 'emotion_model.h5'
print(f"Loading model from: {MODEL_PATH}")
try:
    emotion_model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    emotion_model = None

def preprocess_audio(filepath):
    # Convert MP3 to WAV if needed
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.mp3':
        sound = AudioSegment.from_mp3(filepath)
        wav_path = filepath.replace('.mp3', '.wav')
        sound.export(wav_path, format='wav')
        filepath = wav_path
    # Load audio file
    y, sr = librosa.load(filepath, sr=22050, mono=True)
    # Extract MFCCs (adjust n_mfcc as needed for your model)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    # Reshape for model input (1, 40)
    return np.expand_dims(mfcc_scaled, axis=0)

def preprocess_audio_from_bytes(audio_bytes, audio_format='wav'):
    """Preprocess audio from bytes (for real-time analysis)"""
    try:
        print(f"Preprocessing audio: {len(audio_bytes)} bytes, format: {audio_format}")
        
        # Create a temporary file to handle the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}') as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        print(f"Created temporary file: {temp_file_path}")
        
        try:
            # Convert to WAV if needed (browser might send webm/mp4)
            if audio_format in ['webm', 'mp4', 'm4a']:
                print(f"Converting {audio_format} to WAV...")
                audio = AudioSegment.from_file(temp_file_path, format=audio_format)
                wav_path = temp_file_path.replace(f'.{audio_format}', '.wav')
                audio.export(wav_path, format='wav')
                temp_file_path = wav_path
                print(f"Converted to WAV: {wav_path}")
            
            # Load audio with librosa
            print(f"Loading audio with librosa: {temp_file_path}")
            y, sr = librosa.load(temp_file_path, sr=22050, mono=True)
            print(f"Audio loaded: {len(y)} samples, {sr} Hz")
            
            # Ensure minimum audio length (at least 1 second)
            if len(y) < sr * 1.0:  # Less than 1 second
                print(f"Audio too short: {len(y)/sr:.2f} seconds")
                return None
            
            print(f"Audio length: {len(y)/sr:.2f} seconds")
            
            # Extract MFCCs
            print("Extracting MFCCs...")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            
            # Reshape for model input (1, 40)
            features = np.expand_dims(mfcc_scaled, axis=0)
            print(f"Features extracted: shape {features.shape}")
            
            return features
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print(f"Cleaned up: {temp_file_path}")
            # Also clean up any converted wav file
            if audio_format in ['webm', 'mp4', 'm4a']:
                wav_path = temp_file_path.replace(f'.{audio_format}', '.wav')
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                    print(f"Cleaned up: {wav_path}")
                    
    except Exception as e:
        import traceback
        print(f"Error preprocessing audio: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio = request.files['audio']
        if audio and audio.filename:
            allowed_extensions = {'.wav', '.mp3'}
            ext = os.path.splitext(audio.filename)[1].lower()
            if ext not in allowed_extensions:
                return render_template('index.html', prediction=None, filename=None, error='Only WAV and MP3 files are allowed.')
            # Save the uploaded file
            filepath = os.path.join('static', audio.filename)
            audio.save(filepath)
            try:
                features = preprocess_audio(filepath)
                prediction = emotion_model.predict(features)
                predicted_label = emotion_labels[np.argmax(prediction)]
            except Exception as e:
                return render_template('index.html', prediction=None, filename=audio.filename, error=f'Error processing audio: {str(e)}')
            return render_template('index.html', prediction=predicted_label, filename=audio.filename)
    return render_template('index.html')

@app.route('/analyze_realtime', methods=['POST'])
def analyze_realtime():
    """Handle real-time audio analysis"""
    try:
        print("=== Starting real-time analysis ===")
        
        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({'success': False, 'error': 'No audio file provided'})
        
        audio_file = request.files['audio']
        
        # Get file extension to determine format
        filename = audio_file.filename or 'audio.wav'
        audio_format = filename.split('.')[-1].lower() if '.' in filename else 'wav'
        
        print(f"Audio file: {filename}, format: {audio_format}")
        
        # Read audio bytes
        audio_bytes = audio_file.read()
        
        if len(audio_bytes) == 0:
            print("Empty audio file received")
            return jsonify({'success': False, 'error': 'Empty audio file'})
        
        print(f"Received audio: {len(audio_bytes)} bytes, format: {audio_format}")
        
        # Preprocess audio
        print("Starting audio preprocessing...")
        features = preprocess_audio_from_bytes(audio_bytes, audio_format)
        
        if features is None:
            print("Audio preprocessing failed - returned None")
            return jsonify({'success': False, 'error': 'Failed to process audio - audio too short or invalid format'})
        
        print(f"Audio preprocessing successful, features shape: {features.shape}")
        
        # Make prediction
        print("Making prediction...")
        if emotion_model is None:
            print("Model not loaded!")
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
        prediction = emotion_model.predict(features, verbose=0)  # Suppress verbose output
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)
        
        print(f"Prediction: {predicted_label}, Confidence: {confidence}%")
        
        return jsonify({
            'success': True,
            'emotion': predicted_label,
            'confidence': round(confidence, 1)
        })
        
    except Exception as e:
        import traceback
        print(f"Error in analyze_realtime: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
