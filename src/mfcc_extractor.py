import librosa
import numpy as np
import os
import pandas as pd

def extract_features(file_path, n_mfcc=13):
    """
    Extracts MFCCs and flattens them into a 1D vector.
    """
    audio, sr = librosa.load(file_path, sr=None) # sr=None keeps the processed 16kHz
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Flatten to 1D: (n_mfcc * n_frames)
    # This turns our "image" of sound into a single long "fingerprint" line
    return mfccs.flatten()

def create_feature_csv(processed_dir, output_csv):
    """
    Loops through processed words and saves everything into one CSV.
    """
    data = []
    
    # Walk through each folder in processed/
    for word_label in os.listdir(processed_dir):
        word_path = os.path.join(processed_dir, word_label)
        if not os.path.isdir(word_path): continue
        
        print(f"Extracting features for: {word_label}")
        
        for file_name in os.listdir(word_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(word_path, file_name)
                
                try:
                    features = extract_features(file_path)
                    # Add features + the label to our list
                    sample_row = list(features) + [word_label]
                    data.append(sample_row)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    # Create Column Names: Feature_1, Feature_2, ..., Label
    num_features = len(data[0]) - 1
    columns = [f"feature_{i}" for i in range(num_features)] + ["label"]
    
    # Save to CSV
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Success! Feature bank saved to {output_csv}")

if __name__ == "__main__":
    create_feature_csv("data/processed", "data/features/malayalam_features.csv")