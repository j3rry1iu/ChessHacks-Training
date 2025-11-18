import subprocess
from huggingface_hub import hf_hub_download

# 1. Download from Hugging Face
print("📉 Downloading from Hugging Face...")
local_path = hf_hub_download(
    repo_id = "HamzaAmmar/chesshacks-model" ,
    filename="best.pt",
    local_dir="weights"
)

# 2. Upload to Modal
print("📈 Uploading to Modal Volume (chessbot-weights)...")
subprocess.run([
    "modal", "volume", "put", 
    "chessbot-weights", 
    local_path, 
    "best.pt"
], check=True)

print("✅ Sync Complete! Modal now has the latest weights.")