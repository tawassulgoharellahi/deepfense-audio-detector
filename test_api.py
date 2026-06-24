import os
import time
import subprocess
import requests
import numpy as np
import soundfile as sf

def main():
    print("=== DeepFense AI Audio Detector API Integration Test ===")
    
    # 1. Start uvicorn server in a subprocess
    server_cmd = ["python3", "app.py"]
    print("Starting server process...")
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to spin up
    print("Waiting 5 seconds for server to initialize model and startup...")
    time.sleep(5)
    
    # Check if server process is still running
    if server_process.poll() is not None:
        print("Error: Server failed to start. Stdout:")
        out, err = server_process.communicate()
        print(out)
        print("Stderr:")
        print(err)
        return
        
    # 2. Create a temporary 3-second audio file
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.randn(len(t))
    test_file = "test_api_temp.wav"
    sf.write(test_file, waveform, sr)
    print(f"Generated temp audio file: {test_file}")
    
    try:
        # 3. Send POST request to FastAPI endpoint
        url = "http://localhost:7860/api/detect"
        print(f"Sending POST request to {url}...")
        
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'audio/wav')}
            response = requests.post(url, files=files)
            
        print("\nResponse Status Code:", response.status_code)
        print("Response JSON Payload:")
        import json
        print(json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"\nAPI test failed with error: {e}")
        
    finally:
        # 4. Cleanup server process
        print("\nStopping server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            print("Server process stopped.")
        except subprocess.TimeoutExpired:
            server_process.kill()
            print("Server process force killed.")
            
        # 5. Cleanup temp audio
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up {test_file}")

if __name__ == "__main__":
    main()
