from huggingface_hub import HfApi

# --- CONFIGURATION ---
REPO_ID = "HamzaAmmar/chesshacks-model"
LOCAL_FILE = "weights/best.pt"
REMOTE_FILENAME = "best.pt"

# YOUR NEW WRITE TOKEN
HF_TOKEN = "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"
# ---------------------

print(f"🚀 Uploading {LOCAL_FILE} to {REPO_ID}...")

api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=LOCAL_FILE,
    path_in_repo=REMOTE_FILENAME,
    repo_id=REPO_ID,
    repo_type="model"
)

print("✅ Upload complete!")