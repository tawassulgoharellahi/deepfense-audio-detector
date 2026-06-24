import urllib.request
import urllib.error

urls = [
    "https://valle.blob.core.windows.net/share/wavlm/WavLM-Large.pt",
    "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt",
]

for url in urls:
    print(f"Testing URL: {url}")
    req = urllib.request.Request(url, method='HEAD')
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"  Status: {resp.status}")
            print(f"  Content-Length: {resp.getheader('Content-Length')}")
    except Exception as e:
        print(f"  Failed: {e}")
