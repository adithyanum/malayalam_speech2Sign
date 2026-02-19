import librosa
import numpy as np
import os
import pandas as pd

def extract_features(file_path, n_mfcc=13):
    """
    Extracts a high-dimensional feature vector using MFCCs and their temporal derivatives.
    
    This method captures:
    1. Static spectral shape (MFCC)
    2. Velocity of spectral change (Delta)
    3. Acceleration of spectral change (Delta-Delta)
    """
    # Load audio; sr=None preserves the original sampling rate (e.g., 16kHz)
    audio, sr = librosa.load(file_path, sr=None) 
    
    # 1. Extract Base MFCCs (The 'Static' fingerprint)
    # Shape: (n_mfcc, n_frames)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # 2. Calculate First-Order Derivatives (Delta)
    # Captures the transition between phonemes, vital for Malayalam syllable structure
    delta_mfcc = librosa.feature.delta(mfcc)
    
    # 3. Calculate Second-Order Derivatives (Delta-Delta)
    # Captures the rhythm and acceleration of speech
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # 4. Feature Stacking (Vertical concatenation)
    # Combining static and dynamic features creates a comprehensive 3D acoustic model
    combined = np.vstack((mfcc, delta_mfcc, delta2_mfcc))
    
    # Flattening to 1D vector (Size: n_mfcc * n_frames * 3)
    # This vector serves as the numerical input for the SVM Classifier
    return combined.flatten()

def create_feature_csv(processed_dir, output_csv):
    """
    Iterates through organized audio directories to generate a structured dataset CSV.
    """
    data = []
    
    if not os.path.exists(processed_dir):
        print(f"Directory not found: {processed_dir}")
        return

    # Iterate through each sub-folder (labels) in the processed data directory
    for word_label in sorted(os.listdir(processed_dir)):
        word_path = os.path.join(processed_dir, word_label)
        if not os.path.isdir(word_path):
            continue
        
        print(f"🔍 Extracting features for label: {word_label}")
        
        for file_name in os.listdir(word_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(word_path, file_name)
                
                try:
                    features = extract_features(file_path)
                    # Append feature vector and its corresponding label
                    sample_row = list(features) + [word_label]
                    data.append(sample_row)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    if not data:
        print("No audio data detected in the specified directory.")
        return

    # Generate dynamic column headers: feature_0, feature_1, ..., label
    num_features = len(data[0]) - 1
    columns = [f"feature_{i}" for i in range(num_features)] + ["label"]
    
    # Construct DataFrame and export to CSV
    df = pd.DataFrame(data, columns=columns)
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Pipeline Complete!")
    print(f"📁 Dataset saved to: {output_csv}")
    print(f"📊 Features per sample: {num_features}")

if __name__ == "__main__":
    # Define paths for raw processed audio and the target feature bank
    INPUT_DIR = "data/processed"
    OUTPUT_FILE = "data/features/malayalam_features.csv"
    
    create_feature_csv(INPUT_DIR, OUTPUT_FILE)