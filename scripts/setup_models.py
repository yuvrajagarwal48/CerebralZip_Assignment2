import os
import shutil
from huggingface_hub import hf_hub_download

os.makedirs("models", exist_ok=True)

models_to_fetch = [
    ("joker5914/yolov8n-license-plate", "best.pt", "models/plate_detector.pt"),
    ("mikestealth/nafnet-models", "NAFNet-GoPro-width32.pth", "models/NAFNet-GoPro-width32.pth"),
    ("ai-forever/Real-ESRGAN", "RealESRGAN_x4.pth", "models/RealESRGAN_x4.pth")
]

for repo_id, filename, target_path in models_to_fetch:
    if os.path.exists(target_path):
        print(f"[✓] {target_path} already exists ({os.path.getsize(target_path) / (1024*1024):.1f} MB).")
        continue
    print(f"[*] Fetching {repo_id}/{filename} -> {target_path}...")
    try:
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(cached_path, target_path)
        print(f"[✓] Successfully stored in {target_path} ({os.path.getsize(target_path) / (1024*1024):.1f} MB).")
    except Exception as e:
        print(f"[!] Warning: Failed to fetch {repo_id}/{filename}: {e}")

