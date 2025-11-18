import modal
import os

# Connect to the SAME volume used in training
app = modal.App("chess-fetch-weights")
volume = modal.Volume.from_name("chessbot-weights")

@app.function(volumes={"/weights": volume})
def get_file_content(filename):
    file_path = f"/weights/{filename}"
    print(f"Looking for {file_path}...")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return f.read()
    return None

@app.function(volumes={"/weights": volume})
def list_files():
    return os.listdir("/weights")

@app.local_entrypoint()
def main():
    # 1. List files to be sure
    print("Files in Modal Volume:")
    files = list_files.remote()
    print(files)

    # 2. Download best.pt
    target_file = "best.pt"
    if target_file in files:
        print(f"\nDownloading {target_file}...")
        data = get_file_content.remote(target_file)
        
        os.makedirs("weights", exist_ok=True)
        local_path = os.path.join("weights", target_file)
        
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"✅ Saved to {local_path}")
    else:
        print(f"\n❌ Could not find {target_file} in the volume.")