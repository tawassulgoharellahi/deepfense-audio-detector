import torch

checkpoint_path = '/Users/tge/Documents/ai_audio_detector/models/ASV19_WavLM_Nes2Net_NoAug_Seed42/best_model.pth'
state_dict = torch.load(checkpoint_path, map_location='cpu')

print("Type of state_dict:", type(state_dict))
for k, v in state_dict.items():
    print(f"Key: {k}, Type: {type(v)}")
    if isinstance(v, dict):
        print(f"  Subkeys (first 10): {list(v.keys())[:10]}")
    elif hasattr(v, 'shape'):
        print(f"  Shape: {v.shape}")
