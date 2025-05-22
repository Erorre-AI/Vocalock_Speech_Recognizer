# 🔐 Vocalock – Voice-Activated Access Control

Vocalock is a secure, speaker-recognition-based access control system that uses **voice biometric authentication**. It combines speaker verification and speech transcription using advanced audio features and OpenAI's Whisper model. 

Built with **Streamlit** for an interactive interface and **PyAudio + Librosa** for voice processing.

---

## 🚀 Features

- 🎙️ Voice Enrollment: Capture and store user voiceprints and passphrases
- 🔑 Voice Authentication: Match voiceprints + passphrases to allow access
- 📜 Access Logs: View daily access attempts with pass/fail status
- 🧠 Whisper Transcription: Converts speech to text to verify passphrases
- ✅ Streamlit Interface: Intuitive web UI for testing and deployment

---


```bash
streamlit run vocalock_app.py

## 🛠️ Installation

### 1. Clone the Repo

```bash
git clone https://github.com/Erorre-AI/Vocalock_Speech_Recognizer.git
cd Vocalock_Speech_Recognizer
```

### 2. Create Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

> For PyAudio on Windows:

```bash
pip install pipwin
pipwin install pyaudio
```

---

## 📁 Project Structure

```
Vocalock_Speech_Recognizer/
│
├── vocalock.py          # Voice authentication system backend
├── vocalock_app.py      # Streamlit interface
├── requirements.txt     # Dependencies
├── enrolled_users/      # Voice data (created after first enrollment)
└── access_logs/         # Daily access log files
```

---

## ⚙️ How It Works

1. **Enroll a user**

   * Records voice and transcribes passphrase
   * Extracts MFCC + pitch + spectral features
   * Saves to disk

2. **Authenticate user**

   * Compares spoken passphrase and voiceprint
   * Checks attempt limits and logs access

3. **Access decision**

   * Displays result in Streamlit UI
   * Logs timestamp, decision, and match scores

---

## ✅ Requirements

* Python 3.8+
* [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
* [Librosa](https://librosa.org)
* [OpenAI Whisper](https://github.com/openai/whisper)
* [Streamlit](https://streamlit.io)

---

## 🧠 Powered By

* OpenAI Whisper for speech-to-text
* Librosa + NumPy for audio feature extraction
* PyAudio for microphone input
* Streamlit for GUI

---

## 📜 License

MIT License © 2025 [Erorre-AI](https://github.com/Erorre-AI)

---

## ✨ Future Improvements

* Voiceprint clustering & anomaly detection
* Web-based admin dashboard
* Support for audio file uploads

```
