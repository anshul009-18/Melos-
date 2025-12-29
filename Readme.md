# 🎵 Melos — Emotion-Aware Music Generation using Hybrid Deep Learning

Melos is a **hybrid deep learning system** for **emotion-aware music generation** that combines the strengths of a **Variational Autoencoder (VAE)** and a **Transformer** to generate emotionally expressive, coherent music clips based on user-selected emotions.

The system is designed as an **end-to-end local application**, featuring intelligent emotion conditioning, rule-based enhancements, and an interactive GUI for real-time music creation.

---

## 📌 Key Features

- 🎼 **Emotion-conditioned music generation** (8 emotions)
- 🧠 **Hybrid VAE + Transformer architecture**
- 🎹 Supports **cinematic piano-style compositions**
- 🎛 **Interactive GUI** for emotion, duration & creativity control
- 🔊 High-quality **audio synthesis from mel spectrograms**
- 🎚 Rule-based music theory enhancements for structure & harmony
- 💻 Fully **local deployment** (no cloud required)

---

## 🎭 Supported Emotions

The system supports **8 discrete emotional categories**:

- Happy  
- Sad  
- Energetic  
- Calm  
- Romantic  
- Angry  
- Nostalgic  
- Mysterious  

Each emotion influences tempo, dynamics, timbre, and musical progression.

---

## 🧠 System Architecture

Melos uses a **two-stage hybrid architecture**:

### 1️⃣ Variational Autoencoder (VAE)
- Learns latent musical representations
- Enables **diverse outputs** for the same emotion
- Prevents mode collapse via KL regularization

### 2️⃣ Emotion-Conditioned Transformer
- Models long-term musical dependencies
- Uses attention mechanisms for coherence
- Conditioned on both **latent vectors** and **emotion embeddings**

### 3️⃣ Rule-Based Music Enhancer
- Adds harmony, chord progressions, dynamics
- Improves musical realism and structure
- Acts as a stabilizing layer over DL outputs

---

## 📂 Datasets Used

### 🎧 Free Music Archive (FMA – Medium)
- 25,000 audio tracks (30s clips)
- Used for large-scale audio training
- Emotion labels inferred using acoustic features

### 🎹 EMOPIA Dataset
- Emotion-labeled piano music
- 4 valence-arousal quadrants
- Used for emotion grounding

### 🎼 MAESTRO Dataset
- High-quality aligned MIDI + audio
- Supports expressive piano modeling

---

## 🔄 Project Workflow

1. **Audio Preprocessing**
   - Load audio (Librosa)
   - Extract MFCCs, tempo, spectral features
   - Generate mel spectrograms

2. **Emotion Labeling**
   - Rule-based inference (tempo, RMS, centroid)
   - Emotion mapped to one-hot embeddings

3. **Model Training**
   - VAE learns latent structure
   - Transformer learns sequential coherence
   - Loss = MSE + KL Divergence

4. **Music Generation**
   - User selects emotion & duration
   - Latent vector sampled
   - Transformer generates mel frames
   - Griffin-Lim reconstructs audio

5. **Post-Processing**
   - Harmony, bass, dynamics
   - Stereo WAV export

---

## 🖥️ User Interface

- Built using **Tkinter**
- Emotion selection via dropdown
- Duration & creativity (temperature) controls
- Real-time playback using **Pygame**
- One-click music generation and export

---

## 📊 Evaluation Metrics

- **Fréchet Audio Distance (FAD):** 0.42 ± 0.08  
- **Emotion Alignment Accuracy:** ~78%  
- **Reconstruction Loss:** MSE + KL Divergence  
- **Sequence Perplexity:** 4.8  

Hybrid conditioning improves emotion alignment by **~45%** compared to unconditioned baselines.

---

## 🛠️ Tech Stack

**Languages**
- Python 3.8+

**Libraries**
- PyTorch
- Librosa
- NumPy, SciPy
- Torchaudio
- Scikit-learn
- Tkinter, Pygame
- SoundFile

**Tools**
- Git
- Jupyter Notebook

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/anshul009-18/Melos-Music.git
cd melos

# Install dependencies
pip install -r requirements.txt

# Run the application
python emotion_music_generator_model.py
