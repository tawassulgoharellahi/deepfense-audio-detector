import os
import requests
import numpy as np
import soundfile as sf

def main():
    print("=== Querying Running DeepFense API Endpoint ===")
    
    # 1. Create a 5-second mock audio file
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.randn(len(t))
    test_file = "test_live_api.wav"
    sf.write(test_file, waveform, sr)
    print(f"Generated temp audio file: {test_file}")
    
    try:
        # 2. Send POST request
        url = "http://localhost:7860/api/detect"
        print(f"Sending POST request to {url}...")
        
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'audio/wav')}
            response = requests.post(url, files=files)
            
        print("\nResponse Status Code:", response.status_code)
        
        if response.status_code == 200:
            import json
            print("Response JSON Payload:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Details:", response.text)
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        
    finally:
        # 3. Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up {test_file}")

if __name__ == "__main__":
    main()
