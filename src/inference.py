import librosa
import numpy as np
import joblib
import sounddevice as sd # For live recording
import soundfile as sf

# 1. Load the "Brain" and the Scalers
model = joblib.load('models/malayalam_svm.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def predict_live_speech():
    # 2. Record 1 second of audio
    fs = 16000
    print("Speak now...")
    recording = sd.rec(int(1.0 * fs), samplerate=fs, channels=1)
    sd.wait()
    
    # 3. Preprocess (Must match Friend 1's logic!)
    # Trim silence
    audio_trimmed, _ = librosa.effects.trim(recording.flatten(), top_db=20)
    
    # Pad to exactly 1.0s (16000 samples)
    target_samples = 16000
    if len(audio_trimmed) < target_samples:
        audio_final = np.pad(audio_trimmed, (0, target_samples - len(audio_trimmed)), mode='constant')
    else:
        audio_final = audio_trimmed[:target_samples]
        
    # 4. Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_final, sr=fs, n_mfcc=13)
    features = mfccs.flatten().reshape(1, -1) # Reshape for a single prediction
    
    # 5. Scale and Predict
    features_scaled = scaler.transform(features)
    prediction_id = model.predict(features_scaled)
    word = label_encoder.inverse_transform(prediction_id)[0]
    
    print(f"Predicted Word: {word}")
    return word

if __name__ == "__main__":
    predict_live_speech()