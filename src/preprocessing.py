import librosa
import numpy as np
import os
import soundfile as sf

def clean_audio(file_path, target_sr=16000, duration=1.0):
    """
    Standardizes a single audio file: Resamples, Trims, and Pads/Truncates.
    """
    # 1. Load and Resample to 16kHz
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # 2. Voice Activity Detection (VAD) - Trim silence
    # top_db=20 is a good starting point for Malayalam phonetics
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    
    # 3. Fixed-Length Padding/Truncating (The 1-Second Rule)
    target_samples = int(duration * target_sr)
    
    if len(audio_trimmed) < target_samples:
        # Pad with silence if too short
        padding = target_samples - len(audio_trimmed)
        audio_final = np.pad(audio_trimmed, (0, padding), mode='constant')
    else:
        # Truncate if too long
        audio_final = audio_trimmed[:target_samples]
        
    # 4. Amplitude Normalization
    audio_final = librosa.util.normalize(audio_final)
    
    return audio_final, target_sr

def process_dataset(input_dir, output_dir):
    """
    Crawl through folders and save cleaned versions.
    """
    for word_folder in os.listdir(input_dir):
        word_path = os.path.join(input_dir, word_folder)
        if not os.path.isdir(word_path): continue
            
        # Create corresponding output folder
        out_word_path = os.path.join(output_dir, word_folder)
        os.makedirs(out_word_path, exist_ok=True)
        
        for file_name in os.listdir(word_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(word_path, file_name)
                
                # Preprocess the file
                cleaned_audio, sr = clean_audio(file_path)
                
                # Save as a clean .wav file
                save_path = os.path.join(out_word_path, f"clean_{file_name}")
                sf.write(save_path, cleaned_audio, sr)
                print(f"Processed: {word_folder}/{file_name}")

if __name__ == "__main__":
    process_dataset("data/raw", "data/processed")