### Malayalam Speech to Indian Sign Language (ISL) Generator

A Machine Learning prototype that acts as an automated interpreter, converting spoken Malayalam words into Indian Sign Language (ISL) animations. This project serves as a technical feasibility study and MVP aimed at bridging the communication gap between the hearing majority and the Deaf and Hard-of-Hearing (DHH) community in Kerala.

## 🚀 Project Overview

While existing solutions often rely on massive datasets and complex deep learning architectures (which are largely unavailable for Malayalam and ISL), this project takes a "Smart Scope" approach. We built an Isolated Word Recognizer tailored for a limited vocabulary.

Given the constrained dataset (6 target words, 30 samples each), we strategically implemented a Support Vector Machine (SVM) to prevent the overfitting common in Neural Networks, paired with MediaPipe for lightweight, real-time sign language skeletal visualization.

## 🛠️ Tech Stack

# Audio Processing: librosa, soundfile, numpy

# Machine Learning: scikit-learn (SVM Classifier, StandardScaler)

# Sign Language Visualization: MediaPipe (Holistic/Pose landmarks), OpenCV (cv2)

# Data Augmentation: Custom Python scripts (Pitch shifting, time stretching, noise injection)

## ⚙️ System Architecture & Workflow

Our pipeline follows a linear Hear $\rightarrow$ Process $\rightarrow$ Classify $\rightarrow$ Act sequence.

# 1. Audio Input & Pre-processing (The Ear)

Capture: Records or loads spoken Malayalam audio (.wav).

Clean: Truncates silence from the beginning and end to isolate the word.

Standardize: Resamples all audio to a standard 16kHz format.

# 2. Feature Extraction (The Bridge)

MFCCs: Converts the raw audio waveform into Mel-Frequency Cepstral Coefficients (MFCCs), creating a numerical representation of human speech phonemes.

Padding/Truncating: Because SVMs require fixed-length inputs, every MFCC matrix is padded with zeros or truncated to match a strict frame length (e.g., 44 frames).

Flattening: The 2D MFCC array is flattened into a 1D vector for the SVM.

# 3. Classification (The Brain)

Model: A Support Vector Machine (SVM) equipped with an RBF (Radial Basis Function) Kernel.

Why SVM? With only 180 total raw samples, Deep Learning (MLP/CNN) would severely overfit. The SVM mathematically calculates the optimal decision boundaries between our 6 word classes with high efficiency.

# 4. Visual Output (The Body)

Dictionary Lookup: The SVM outputs a class ID (e.g., Class 0), which maps to a word (e.g., "Namaskaram").

Landmark Retrieval: The system retrieves a pre-recorded .csv file containing structural frame-by-frame coordinate data for that specific sign.

MediaPipe Rendering: OpenCV reads the CSV and continuously plots the MediaPipe skeletal landmarks on a black canvas or avatar background to animate the ISL gesture.

## 📊 Dataset & Augmentation Strategy

To train a robust model without thousands of real human samples, we employed Data Augmentation:

# Base Data: 6 Words $\times$ 30 Samples/Word = 180 Raw Samples.

# Augmentation Pipeline: We cloned the raw data by programmatically applying random effects:

White Noise Injection (Simulating background environments)

Pitch Shifting ($\pm$ 2 semitones to simulate high/low voices)

Speed Variations ($\pm$ 10% speed)

# Final Training Data: Scaled up to 300 samples per word (1,800 total samples), significantly improving the SVM's real-world accuracy.

## 💻 Installation & Usage

Prerequisites

Python 3.8+

A working microphone (for live inference)

Setup

Clone the repository:

git clone [https://github.com/adithyanum/malayalam_speech2Sign.git](https://github.com/adithyanum/malayalam_speech2Sign.git)
cd malayalam_speech2Sign


Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


(Note: Add specific execution commands here, e.g., python src/train_svm.py or python main.py based on your final file structure).

## 🔮 Future Roadmap

This MVP lays the groundwork for a continuous speech-to-sign language system. Future scaling will involve:

Continuous Speech Recognition: Replacing the SVM with OpenAI Whisper or IndicWav2Vec to transcribe full Malayalam sentences in real-time.

NLP & Grammar Translation: Converting Malayalam Spoken Grammar (SOV) into visual-spatial ISL Glosses (e.g., "Enikku Vellam Venam" $\rightarrow$ "ME WATER WANT").

3D Avatar Generation: Moving beyond MediaPipe skeletons to a fully rigged 3D Avatar built in Unity 3D or Three.js, implementing animation blending to smoothly transition between words.

Domain Restriction: Limiting vocabulary to high-impact scenarios (e.g., Hospital Receptions, Railway Stations) to maintain high accuracy and realism.

## 👥 Team

# Aathithya

# Akhilesh 

# Adithyan



Built to make communication accessible to all.
