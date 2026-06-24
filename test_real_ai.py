import os
from utils import preprocess_audio
from detector import AudioDeepfakeDetector

def test_file(detector, path, label):
    print(f"\n--- Testing {label} File: {path} ---")
    if not os.path.exists(path):
        print("File does not exist!")
        return
        
    try:
        preprocessed = preprocess_audio(path)
        print(f"Preprocessed audio shape: {preprocessed.shape} (duration: {len(preprocessed)/16000:.2f}s)")
        
        # Run raw forward pass to see logits and probs
        import torch
        # Let's chunk it just like detector does
        chunk_size = 64000
        chunks = []
        start = 0
        while start < len(preprocessed):
            end = start + chunk_size
            chunk = preprocessed[start:end]
            if len(chunk) < chunk_size:
                chunk = detector.pad_waveform(chunk, chunk_size)
            chunks.append(chunk)
            start += chunk_size
            
        batch = torch.tensor(chunks, dtype=torch.float32).to(detector.device)
        with torch.no_grad():
            outputs = detector.model(batch)
            logits = outputs['logits'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            probs = outputs['probs'].cpu().numpy()
            
        print("Raw Model Outputs:")
        for i in range(len(chunks)):
            print(f"  Segment {i}:")
            print(f"    Logits (0=spoof, 1=bonafide): {logits[i]}")
            print(f"    Score (bonafide logit): {scores[i]}")
            print(f"    Prob (bonafide prob): {probs[i]}")
            print(f"    Spoof Prob (1 - bonafide prob): {1.0 - probs[i]}")
            
        # Predict result
        result = detector.predict(preprocessed)
        print("Detector Result:")
        print(f"  Overall Label: {result['overall_label']}")
        print(f"  Spoof Confidence: {result['spoof_confidence']:.4f}")
        print(f"  Real Confidence: {result['real_confidence']:.4f}")
        
    except Exception as e:
        print(f"Error testing file: {e}")
        import traceback
        traceback.print_exc()

def main():
    detector = AudioDeepfakeDetector()
    
    test_file(detector, "ai/Hello and welcome Th.mp3", "AI/Fake")
    test_file(detector, "real/barackobamainauguraladdressarxe_k6SEk274.mp3", "Real/Bonafide")

if __name__ == "__main__":
    main()
