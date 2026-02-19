import librosa
import numpy as np
import joblib
import sounddevice as sd
import sys
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load the optimized model pipeline
try:
    model = joblib.load('models/malayalam_svm.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    
    selector = joblib.load('models/selector.pkl') 

    feature_names = [f"feature_{i}" for i in range(1248)]
except FileNotFoundError:
    print("CRITICAL ERROR: Model components not found in 'models/'. Ensure training is complete.")
    sys.exit()

def get_prediction():
    fs = 16000
    duration = 1.2  
    
    input("\n🎤 Press [ENTER] to start recording...")
    
    print("🔴 Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("✅ Recording captured. Analyzing...")
    
    audio = recording.flatten()

    # --- GATE 1: ENERGY THRESHOLD (VAD) ---
    # Root Mean Square (RMS) check to filter out ambient silence
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.01: 
        return "SILENCE", 0.0

    # --- PREPROCESSING ---
    # Top_db=20 removes silent trailing/leading edges
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    
    # Ensuring fixed-length input (1.0s) for consistent feature vector size
    target_samples = 16000
    if len(audio_trimmed) < target_samples:
        audio_final = np.pad(audio_trimmed, (0, target_samples - len(audio_trimmed)), mode='constant')
    else:
        audio_final = audio_trimmed[:target_samples]
        
    # --- FEATURE EXTRACTION (The 'V2' Pipeline) ---
    # Capturing Static, Velocity, and Acceleration cues
    mfcc = librosa.feature.mfcc(y=audio_final, sr=fs, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack and flatten into a 1,248-dim vector
    combined = np.vstack((mfcc, delta, delta2))
    features = combined.flatten().reshape(1, -1)
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # --- PIPELINE TRANSFORMATIONS ---
    # 1. Scale based on training distributions
    features_scaled = scaler.transform(features_df)
    # 2. Select the 'Elite 300' features identified during training
    features_selected = selector.transform(features_scaled)

    # --- GATE 2: PROBABILISTIC CONFIDENCE GATE ---
    # We use predict_proba to implement a 'Softmax-like' rejection threshold
    probs = model.predict_proba(features_selected)[0]
    max_prob = np.max(probs)
    
    # If the prediction confidence is low, we classify as UNKNOWN to prevent hallucinations
    if max_prob < 0.60: 
        return "UNKNOWN", max_prob
    
    class_idx = np.argmax(probs)
    word = label_encoder.inverse_transform([class_idx])[0]
    return word, max_prob

if __name__ == "__main__":
    print("==============================================")
    print("   MALAYALAM REAL-TIME SPEECH RECOGNITION     ")
    print("==============================================")
    print(f"Supported Words: {', '.join(label_encoder.classes_)}")
    
    try:
        while True:
            word, confidence = get_prediction()
            
            if word == "SILENCE":
                print("⚠️  Status: Signal too weak. Please speak clearly.")
            elif word == "UNKNOWN":
                print(f"❓ Status: Ambiguous Input (Confidence: {confidence:.2f})")
            else:
                print(f"✨ Result: {word}")
                print(f"📈 Confidence: {int(confidence * 100)}%")

            # User control loop
            choice = input("\n[ENTER] to record again | [Q] to quit: ").lower()
            if choice == 'q':
                break


    except KeyboardInterrupt:
        print("\n\nSystem Shutdown. Goodbye!")
