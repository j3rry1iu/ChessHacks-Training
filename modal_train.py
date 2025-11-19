import subprocess
import modal
import sys
import os

app = modal.App("chess-elo2200-train")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "datasets",
        "numpy",
        "python-chess",
        "torch==2.4.1+cu124",
        "torchvision==0.19.1+cu124",
        "torchaudio==2.4.1+cu124",
        extra_index_url="https://download.pytorch.org/whl/cu124"
    )
    .add_local_dir(".", "/root/app")
)

weights_vol = modal.Volume.from_name("chess-bot-weights", create_if_missing=True)
volume = modal.Volume.from_name("chessbot-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 24,
    #volumes={"/root/app/weights": weights_vol, "/weights": volume},
    volumes={"/weights": volume},
    env={"MODAL_WEIGHTS_PATH": "/weights"},
    #secrets=[modal.Secret.from_name("huggingface-secret-2")]
)

def run_training():
    project_root = "/root/app"
    os.chdir(project_root)
    os.getenv("HF_TOKEN")
    print("[Modal] CWD contents: ", os.listdir("."))
    print("[Modal] Starting training via: python src/train.py")

    subprocess.run(["python", "src/train.py"], check=True)
    print("[Modal] Training complete.")

@app.local_entrypoint()
def main():
    run_training.remote()
