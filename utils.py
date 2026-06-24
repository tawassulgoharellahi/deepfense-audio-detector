import os
import librosa
import soundfile as sf
import numpy as np

def get_audio_duration(file_path: str) -> float:
    """Return the duration of the audio file in seconds without loading it fully in memory."""
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception:
        # Fallback to librosa if soundfile fails
        try:
            return librosa.get_duration(path=file_path)
        except Exception as e:
            raise ValueError(f"Could not read audio file: {e}")

def preprocess_audio(file_path: str, target_sr: int = 16000, max_duration: float = 12.0) -> np.ndarray:
    """
    Validates, loads, resamples, and cleans the input audio file:
    1. Validates duration is <= max_duration (12 seconds).
    2. Loads and resamples to target_sr (default 16kHz) and mono.
    3. Trims silence from start/end.
    """
    # 1. Validate duration
    duration = get_audio_duration(file_path)
    if duration > max_duration:
        raise ValueError(f"Audio file duration ({duration:.2f}s) exceeds the maximum limit of {max_duration:.1f} seconds.")
    
    # 2. Load and resample
    audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # 3. Trim silence (top_db=25 is a standard threshold for voice)
    # We ensure we don't trim to empty audio
    if len(audio_data) > 0:
        trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=25)
        if len(trimmed_audio) > 0:
            audio_data = trimmed_audio
            
    return audio_data
