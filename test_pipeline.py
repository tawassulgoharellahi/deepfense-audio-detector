import os
import numpy as np
import soundfile as sf
from utils import preprocess_audio
from detector import AudioDeepfakeDetector

def main():
    print("=== DeepFense AI Audio Detector Test Pipeline ===")
    
    # 1. Generate a mock audio file: 6 seconds of a 440Hz sine wave (sr=16000)
    sr = 16000
    duration = 6.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate a sine wave + some low noise
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * np.random.randn(len(t))
    mock_waveform = sine_wave + noise
    
    test_file = "test_temp.wav"
    sf.write(test_file, mock_waveform, sr)
    print(f"Generated mock audio file: {test_file} ({duration} seconds)")
    
    try:
        # 2. Run preprocessing
        print("\nPreprocessing audio...")
        preprocessed = preprocess_audio(test_file, target_sr=sr, max_duration=12.0)
        print(f"Preprocessed audio shape: {preprocessed.shape} (duration: {len(preprocessed)/sr:.2f}s)")
        
        # 3. Initialize detector
        print("\nLoading detector...")
        detector = AudioDeepfakeDetector()
        
        # 4. Predict
        print("\nRunning inference...")
        result = detector.predict(preprocessed, sample_rate=sr)
        
        print("\nInference Result:")
        import json
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"\nPipeline verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 5. Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nCleaned up temporary test file: {test_file}")

if __name__ == "__main__":
    main()
