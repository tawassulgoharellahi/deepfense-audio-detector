import yaml
import torch
import numpy as np
from deepfense.models.detector import ModularDetector

config_path = '/Users/tge/Documents/ai_audio_detector/models/ASV19_WavLM_Nes2Net_NoAug_Seed42/config.yaml'
checkpoint_path = '/Users/tge/Documents/ai_audio_detector/models/ASV19_WavLM_Nes2Net_NoAug_Seed42/best_model.pth'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update checkpoint path to local
config['model']['frontend']['args']['ckpt_path'] = '/Users/tge/Documents/ai_audio_detector/models/WavLM-Large.pt'

detector = ModularDetector(config['model'])
state_dict = torch.load(checkpoint_path, map_location='cpu')
detector.load_state_dict(state_dict['model_state'])
detector.eval()

print("ModularDetector loaded successfully!")

# Create a mock 4-second audio signal at 16kHz (64,000 samples)
mock_input = torch.randn(1, 64000)

with torch.no_grad():
    output = detector(mock_input)
    print("Output type:", type(output))
    if isinstance(output, tuple):
        print("Output is a tuple of length:", len(output))
        for i, val in enumerate(output):
            print(f"  Element {i}: type {type(val)}, shape/value {getattr(val, 'shape', val)}")
    elif isinstance(output, dict):
        print("Output is a dict. Keys:")
        for k, v in output.items():
            print(f"  {k}: type {type(v)}, shape/value {getattr(v, 'shape', v)}")
    else:
        print("Output shape/value:", getattr(output, 'shape', output))
