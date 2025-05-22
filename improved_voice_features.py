import numpy as np
import librosa
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import warnings
warnings.filterwarnings('ignore')

def extract_advanced_voice_features(audio_file, sample_rate=16000):
    """
    Extract advanced voice features for speaker verification.

    Args:
        audio_file: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Dictionary containing various voice features
    """
    # Load audio using librosa for better processing
    try:
        y, sr = librosa.load(audio_file, sr=sample_rate)
    except:
        # Fallback to pyAudioAnalysis
        [sr, y] = audioBasicIO.read_audio_file(audio_file)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        if sr != sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate

    # Remove silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    if len(y_trimmed) == 0:
        y_trimmed = y

    features = {}

    # 1. MFCC Features (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    features['mfcc_delta'] = np.mean(librosa.feature.delta(mfccs), axis=1)

    # 2. Pitch/F0 Features (Fundamental frequency)
    f0, voiced_flag, voiced_probs = librosa.pyin(y_trimmed,
                                                fmin=librosa.note_to_hz('C2'),
                                                fmax=librosa.note_to_hz('C7'))
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) > 0:
        features['f0_mean'] = np.mean(f0_clean)
        features['f0_std'] = np.std(f0_clean)
        features['f0_median'] = np.median(f0_clean)
        features['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
        features['voiced_ratio'] = np.sum(voiced_flag) / len(voiced_flag)
    else:
        features['f0_mean'] = 0
        features['f0_std'] = 0
        features['f0_median'] = 0
        features['f0_range'] = 0
        features['voiced_ratio'] = 0

    # 3. Formant-like features (Spectral characteristics)
    spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)[0]

    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

    # 4. Chroma features (Pitch class profiles)
    chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    features['chroma_std'] = np.std(chroma, axis=1)

    # 5. Zero Crossing Rate (Voice quality indicator)
    zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # 6. Tempo and rhythm features
    tempo, beats = librosa.beat.beat_track(y=y_trimmed, sr=sr)
    features['tempo'] = tempo

    # 7. Spectral contrast (Timbral texture)
    spectral_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
    features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1)

    # 8. RMS Energy
    rms = librosa.feature.rms(y=y_trimmed)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    # Convert all features to a single vector
    feature_vector = []
    for key in sorted(features.keys()):
        if isinstance(features[key], np.ndarray):
            feature_vector.extend(features[key].flatten())
        else:
            feature_vector.append(features[key])

    return np.array(feature_vector), features

def compare_voice_features(enrolled_features, auth_features, detailed=True):
    """
    Advanced comparison of voice features with multiple metrics.

    Args:
        enrolled_features: Feature vector from enrollment
        auth_features: Feature vector from authentication
        detailed: Whether to return detailed comparison

    Returns:
        Dictionary with comparison results
    """
    # Ensure same dimensions
    min_len = min(len(enrolled_features), len(auth_features))
    enrolled_features = enrolled_features[:min_len]
    auth_features = auth_features[:min_len]

    # Handle zero vectors
    if np.all(enrolled_features == 0) or np.all(auth_features == 0):
        return {
            'cosine_similarity': 1.0,  # Worst case
            'euclidean_distance': float('inf'),
            'correlation': 0.0,
            'match_probability': 0.0,
            'is_match': False
        }

    # 1. Cosine Similarity (lower is better, 0 = identical)
    cosine_sim = cosine(enrolled_features, auth_features)

    # 2. Euclidean Distance (lower is better)
    euclidean_dist = euclidean(enrolled_features, auth_features)

    # 3. Pearson Correlation (higher is better, 1 = perfect correlation)
    correlation, _ = stats.pearsonr(enrolled_features, auth_features)
    if np.isnan(correlation):
        correlation = 0

    # 4. Normalized features comparison
    enrolled_norm = (enrolled_features - np.mean(enrolled_features)) / (np.std(enrolled_features) + 1e-10)
    auth_norm = (auth_features - np.mean(auth_features)) / (np.std(auth_features) + 1e-10)

    normalized_cosine = cosine(enrolled_norm, auth_norm)

    # 5. Calculate match probability using multiple metrics
    # Weights for different metrics (tuned for voice verification)
    cosine_score = max(0, 1 - cosine_sim)  # Convert to similarity score
    correlation_score = max(0, correlation)
    normalized_score = max(0, 1 - normalized_cosine)

    # Weighted average
    match_probability = (0.4 * cosine_score +
                        0.3 * correlation_score +
                        0.3 * normalized_score)

    # Stricter thresholds for voice matching
    is_match = (cosine_sim < 0.15 and  # Very strict cosine similarity
                correlation > 0.7 and   # High correlation required
                normalized_cosine < 0.2)  # Strict normalized similarity

    results = {
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist,
        'correlation': correlation,
        'normalized_cosine': normalized_cosine,
        'match_probability': match_probability,
        'is_match': is_match
    }

    if detailed:
        print(f"Voice Comparison Results:")
        print(f"  Cosine Similarity: {cosine_sim:.4f} (threshold: < 0.15)")
        print(f"  Correlation: {correlation:.4f} (threshold: > 0.7)")
        print(f"  Normalized Cosine: {normalized_cosine:.4f} (threshold: < 0.2)")
        print(f"  Match Probability: {match_probability:.4f}")
        print(f"  Voice Match: {'YES' if is_match else 'NO'}")

    return results

def extract_speaker_embedding(audio_file):
    """
    Extract a speaker embedding using advanced techniques.
    This is a simplified version - in production, you'd use models like x-vector or d-vector.

    Args:
        audio_file: Path to audio file

    Returns:
        Speaker embedding vector
    """
    feature_vector, feature_dict = extract_advanced_voice_features(audio_file)

    # Focus on speaker-specific features
    speaker_features = []

    # F0 statistics (pitch characteristics)
    speaker_features.extend([
        feature_dict['f0_mean'],
        feature_dict['f0_std'],
        feature_dict['f0_median'],
        feature_dict['f0_range'],
        feature_dict['voiced_ratio']
    ])

    # Spectral characteristics
    speaker_features.extend([
        feature_dict['spectral_centroid_mean'],
        feature_dict['spectral_rolloff_mean'],
        feature_dict['spectral_bandwidth_mean']
    ])

    # MFCC statistics (vocal tract characteristics)
    speaker_features.extend(feature_dict['mfcc_mean'][:5])  # First 5 MFCCs
    speaker_features.extend(feature_dict['mfcc_std'][:5])   # Their standard deviations

    # Voice quality features
    speaker_features.extend([
        feature_dict['zcr_mean'],
        feature_dict['rms_mean']
    ])

    return np.array(speaker_features)

# Test function
def test_voice_robustness(enrolled_audio, test_audio):
    """
    Test the robustness of voice features between two audio files.

    Args:
        enrolled_audio: Path to enrollment audio
        test_audio: Path to test audio

    Returns:
        Detailed comparison results
    """
    print("Extracting features from enrollment audio...")
    enrolled_embedding = extract_speaker_embedding(enrolled_audio)

    print("Extracting features from test audio...")
    test_embedding = extract_speaker_embedding(test_audio)

    print("\nComparing voice features...")
    results = compare_voice_features(enrolled_embedding, test_embedding, detailed=True)

    return results

if __name__ == "__main__":
    print("Advanced Voice Feature Extraction Test")
    print("=====================================")

    # For testing, you would need two audio files
    enrolled_file = input("Enter path to enrolled audio file: ")
    test_file = input("Enter path to test audio file: ")

    if os.path.exists(enrolled_file) and os.path.exists(test_file):
        results = test_voice_robustness(enrolled_file, test_file)
    else:
        print("Audio files not found!")