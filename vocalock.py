# vocalock.py
import os
import json
import time
import pickle
import pyaudio
import wave
import numpy as np
import whisper
import librosa
import warnings
from datetime import datetime
import copy

warnings.filterwarnings('ignore')

LOG_DIR = "../Vocalock_Speech/access_logs"
MAX_ATTEMPTS = 3
COOLDOWN_MINUTES = 5

def safe_json(obj):
    """Convert all unserializable objects in a dict to strings"""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return json.loads(json.dumps(obj, default=str))

def record_audio(filename, duration=5, sample_rate=16000, channels=1, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels,
                    rate=sample_rate, input=True, frames_per_buffer=chunk)
    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename


def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"].strip().lower()


def extract_speaker_embedding(audio_file, sample_rate=16000):
    try:
        y, sr = librosa.load(audio_file, sr=sample_rate)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) == 0:
            y_trimmed = y
    except:
        return np.array([])

    features = []
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfccs)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))

    f0, voiced_flag, _ = librosa.pyin(
        y_trimmed, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) > 0:
        features.extend([np.mean(f0_clean), np.std(f0_clean),
                         np.median(f0_clean), np.max(f0_clean) - np.min(f0_clean),
                         np.sum(voiced_flag) / len(voiced_flag)])
    else:
        features.extend([0, 0, 0, 0, 0])

    features.extend([
        np.mean(librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]),
        np.mean(librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)[0]),
        np.mean(librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]),
        np.mean(librosa.feature.zero_crossing_rate(y_trimmed)[0]),
        np.mean(librosa.feature.rms(y=y_trimmed)[0])
    ])

    return np.array(features)


def normalize_vector(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-10)


def compare_voice_features(enrolled, auth, threshold=0.99):
    if len(enrolled) == 0 or len(auth) == 0:
        return {'cosine_similarity': 0.0, 'is_match': False}
    enrolled_norm = normalize_vector(enrolled)
    auth_norm = normalize_vector(auth)
    cosine_similarity = np.dot(enrolled_norm, auth_norm) / (
        np.linalg.norm(enrolled_norm) * np.linalg.norm(auth_norm))
    is_match = cosine_similarity >= threshold
    return {'cosine_similarity': cosine_similarity, 'is_match': is_match}


def compare_passphrases(enrolled, auth, threshold=0.8):
    if enrolled == auth:
        return True
    enrolled_words = set(enrolled.split())
    auth_words = set(auth.split())
    if not enrolled_words or not auth_words:
        return False
    intersection = enrolled_words.intersection(auth_words)
    union = enrolled_words.union(auth_words)
    similarity = len(intersection) / len(union)
    return similarity >= threshold


def save_enrollment_data(user_id, features, passphrase):
    os.makedirs("../Vocalock_Speech/enrolled_users", exist_ok=True)
    with open(f"../Vocalock_Speech/enrolled_users/{user_id}_data.json", 'w') as f:
        json.dump({"user_id": user_id, "passphrase": passphrase,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    with open(f"../Vocalock_Speech/enrolled_users/{user_id}_features.pkl", 'wb') as f:
        pickle.dump(features, f)


def load_enrollment_data(user_id):
    try:
        with open(f"../Vocalock_Speech/enrolled_users/{user_id}_data.json", 'r') as f:
            passphrase = json.load(f)["passphrase"]
        with open(f"../Vocalock_Speech/enrolled_users/{user_id}_features.pkl", 'rb') as f:
            features = pickle.load(f)
        return passphrase, features
    except:
        return None, None


def enroll_user(user_id):
    audio_file = record_audio(f"{user_id}_enroll.wav")
    features = extract_speaker_embedding(audio_file)
    if len(features) == 0:
        return None, None, None
    passphrase = transcribe_audio(audio_file)
    save_enrollment_data(user_id, features, passphrase)
    return user_id, passphrase, features


def authenticate_user(user_id):
    enrolled_phrase, enrolled_features = load_enrollment_data(user_id)
    if enrolled_phrase is None:
        return False, {"error": "User not enrolled"}
    auth_audio_file = record_audio(f"{user_id}_auth.wav")
    auth_features = extract_speaker_embedding(auth_audio_file)
    if len(auth_features) == 0:
        return False, {"error": "Feature extraction failed"}
    auth_phrase = transcribe_audio(auth_audio_file)
    phrase_match = compare_passphrases(enrolled_phrase, auth_phrase)
    voice_result = compare_voice_features(enrolled_features, auth_features)
    is_match = phrase_match and voice_result["is_match"]
    result = {
        "user_id": user_id,
        "auth_phrase": auth_phrase,
        "enrolled_phrase": enrolled_phrase,
        "phrase_match": phrase_match,
        "voice_similarity": voice_result["cosine_similarity"],
        "voice_match": voice_result["is_match"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return is_match, result


def process_access_request(auth_result, auth_details):
    os.makedirs(LOG_DIR, exist_ok=True)
    user_id = auth_details["user_id"]
    attempts_file = os.path.join(LOG_DIR, "access_attempts.json")
    if not os.path.exists(attempts_file):
        with open(attempts_file, 'w') as f:
            json.dump({}, f)

    with open(attempts_file, 'r') as f:
        data = json.load(f)

    if user_id not in data:
        data[user_id] = {"failed": 0, "cooldown_until": None}

    now = time.time()
    cooldown = data[user_id]["cooldown_until"]

    if cooldown and now < cooldown:
        return False, f"Account locked. Try after {int((cooldown - now) // 60)}m."

    if auth_result:
        data[user_id] = {"failed": 0, "cooldown_until": None}
        access_granted = True
    else:
        data[user_id]["failed"] += 1
        if data[user_id]["failed"] >= MAX_ATTEMPTS:
            data[user_id]["cooldown_until"] = now + COOLDOWN_MINUTES * 60
        access_granted = False

    with open(attempts_file, 'w') as f:
        json.dump(data, f)

    log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}_access_log.json")
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                pass
    logs.append({
        "user_id": user_id,
        "timestamp": auth_details["timestamp"],
        "access_granted": access_granted,
        "details": safe_json(auth_details)
    })


    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    msg = "Access granted." if access_granted else "Access denied."
    return access_granted, msg
