import os
import yaml
import torch
import numpy as np
from deepfense.models.detector import ModularDetector

class AudioDeepfakeDetector:
    def __init__(
        self,
        config_path: str = None,
        checkpoint_path: str = None,
        wavlm_ckpt_path: str = None,
        device: str = 'cpu'
    ):
        self.device = device
        
        # Resolve paths relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            config_path = os.path.join(base_dir, 'models', 'ASV19_WavLM_Nes2Net_NoAug_Seed42', 'config.yaml')
        if checkpoint_path is None:
            checkpoint_path = os.path.join(base_dir, 'models', 'ASV19_WavLM_Nes2Net_NoAug_Seed42', 'best_model.pth')
        if wavlm_ckpt_path is None:
            wavlm_ckpt_path = os.path.join(base_dir, 'models', 'WavLM-Large.pt')
        
        # Download DeepFense model if it doesn't exist locally
        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            print("Model config or checkpoint not found locally. Downloading from Hugging Face Hub...")
            try:
                from deepfense.hub import download_model
                models_dir = os.path.join(base_dir, 'models')
                download_model("ASV19_WavLM_Nes2Net_NoAug_Seed42", output_dir=models_dir)
            except Exception as e:
                print(f"Error downloading DeepFense model: {e}")
                
        # Download WavLM-Large.pt if it doesn't exist locally
        if not os.path.exists(wavlm_ckpt_path):
            print(f"WavLM-Large.pt not found at: {wavlm_ckpt_path}")
            print("Downloading WavLM-Large.pt from Hugging Face Hub...")
            try:
                from huggingface_hub import hf_hub_download
                cached_wavlm_path = hf_hub_download(repo_id="s3prl/converted_ckpts", filename="wavlm_large.pt")
                print(f"WavLM-Large.pt loaded from cache: {cached_wavlm_path}")
                wavlm_ckpt_path = cached_wavlm_path
            except Exception as e:
                print(f"Error downloading WavLM-Large.pt from Hub: {e}")
        
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Dynamically override the frontend WavLM path to point to local file
        self.config['model']['frontend']['args']['ckpt_path'] = wavlm_ckpt_path
        
        # Initialize detector
        print("Initializing ModularDetector model...")
        self.model = ModularDetector(self.config['model'])
        
        # Load weights
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
            
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model_state'])
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded and ready on device:", self.device)

    def pad_waveform(self, x: np.ndarray, max_len: int = 64000) -> np.ndarray:
        """Repeat pad waveform if it's shorter than max_len."""
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        repeats = int(np.ceil(max_len / x_len))
        return np.tile(x, repeats)[:max_len]

    def predict(self, audio_data: np.ndarray, sample_rate: int = 16000, calibration_offset: float = 13.0) -> dict:
        """
        Runs inference on audio data.
        1. Globally normalizes the waveform (zero-mean, unit-variance).
        2. Divides the audio into 4-second chunks (64,000 samples at 16kHz).
        3. Computes the Log-Likelihood Ratio (LLR): logit_bonafide - logit_spoof.
        4. Applies a calibrated sigmoid shift to map the LLR to a probability.
        """
        total_samples = len(audio_data)
        if total_samples == 0:
            raise ValueError("Input audio data is empty after preprocessing.")
            
        # 1. Zero-mean, unit-variance waveform normalization (critical for WavLM alignment)
        mean = audio_data.mean()
        std = audio_data.std() + 1e-5
        audio_data = (audio_data - mean) / std
        
        # 2. Generate chunks
        chunk_size = 64000  # 4 seconds at 16kHz
        chunks = []
        chunk_ranges = []
        
        start = 0
        while start < total_samples:
            end = start + chunk_size
            chunk = audio_data[start:end]
            
            # If it's the last chunk and it's shorter, pad it
            if len(chunk) < chunk_size:
                chunk = self.pad_waveform(chunk, chunk_size)
                
            chunks.append(chunk)
            chunk_ranges.append((start / sample_rate, min(end, total_samples) / sample_rate))
            start += chunk_size
            
            if start >= total_samples:
                break
                
        # Stack chunks into batch
        batch = torch.tensor(np.array(chunks), dtype=torch.float32).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(batch)
            # Extract raw logits: shape (num_chunks, 2)
            logits = outputs['logits'].cpu().numpy()
            
        # 3. Post-process logits using Log-Likelihood Ratio (LLR) and calibration
        segment_results = []
        max_spoof_prob = 0.0
        
        for i, (logit, (start_sec, end_sec)) in enumerate(zip(logits, chunk_ranges)):
            # score = logit_bonafide (index 1) - logit_spoof (index 0)
            score = float(logit[1] - logit[0])
            
            # Apply calibration offset so that 50% probability corresponds to score = -calibration_offset
            # prob_real = sigmoid(score + calibration_offset)
            prob_real = 1.0 / (1.0 + np.exp(-(score + calibration_offset)))
            prob_spoof = 1.0 - prob_real
            
            # Record max spoof probability
            if prob_spoof > max_spoof_prob:
                max_spoof_prob = prob_spoof
                
            label = "Fake/AI" if prob_spoof >= 0.5 else "Real"
            segment_results.append({
                "segment_index": i,
                "time_range": f"{start_sec:.1f}s - {end_sec:.1f}s",
                "real_probability": prob_real,
                "spoof_probability": prob_spoof,
                "label": label,
                "raw_llr_score": score
            })
            
        # Overall assessment
        # The audio is classified as Spoof/Fake if ANY segment has spoof prob >= 0.5 (i.e. LLR <= -calibration_offset)
        is_spoof = bool(max_spoof_prob >= 0.5)
        overall_label = "Fake/AI" if is_spoof else "Real"
        
        # We can compute average spoof prob too
        avg_spoof_prob = float(np.mean([seg["spoof_probability"] for seg in segment_results]))
        max_spoof_prob = float(max_spoof_prob)
        
        return {
            "is_spoof": is_spoof,
            "overall_label": overall_label,
            "spoof_confidence": max_spoof_prob,
            "real_confidence": 1.0 - max_spoof_prob,
            "average_spoof_confidence": avg_spoof_prob,
            "segments": segment_results
        }


