import yaml
import torch
from deepfense.models.detector import ModularDetector

config_path = '/Users/tge/Documents/ai_audio_detector/models/ASV19_WavLM_Nes2Net_NoAug_Seed42/config.yaml'
checkpoint_path = '/Users/tge/Documents/ai_audio_detector/models/ASV19_WavLM_Nes2Net_NoAug_Seed42/best_model.pth'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("Config:")
print(config)

# Instantiate ModularDetector
# Let's see if the config['model'] contains what ModularDetector expects.
model_config = config['model']
detector = ModularDetector(model_config)
print("ModularDetector initialized successfully!")

# Let's load the checkpoint
# WavLM Large uses a lot of weights, let's load it on CPU
state_dict = torch.load(checkpoint_path, map_location='cpu')
print("State dict keys:", list(state_dict.keys())[:10])

# Since state_dict might contain nested keys (like 'state_dict' or 'model'), let's check
if 'state_dict' in state_dict:
    detector.load_state_dict(state_dict['state_dict'])
else:
    detector.load_state_dict(state_dict)
print("Loaded state dict successfully!")
