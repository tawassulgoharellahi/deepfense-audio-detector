import urllib.request
import os
import sys

url = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt"
dest = "/Users/tge/Documents/ai_audio_detector/models/WavLM-Large.pt"

os.makedirs(os.path.dirname(dest), exist_ok=True)

print(f"Downloading {url} to {dest}...")

def progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading... {percent}% ({count * block_size}/{total_size} bytes)")
    sys.stdout.flush()

try:
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print("\nDownload completed successfully!")
except Exception as e:
    print(f"\nError downloading: {e}")
