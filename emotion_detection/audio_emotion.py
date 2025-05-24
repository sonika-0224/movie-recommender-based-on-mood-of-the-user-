import librosa
import numpy as np
import joblib

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    return mfccs

def predict_audio_emotion(file_path):
    model = joblib.load('models/audio_emotion.pkl')
    mfccs = extract_features(file_path)
    return model.predict([mfccs])[0]
