import modal


app = modal.App("chesshacks-training")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "python-chess",
        "datasets",
        "numpy",
    )
    .add_local_dir(".", "/root/app")
)

weights_vol = modal.Volume.from_name("chesshacks-weights", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 3,
    volumes={"/root/app/weights": weights_vol},
)

def run_training():
    import os
    import sys

    project_root = "/root/app"
    sys.path.append(project_root)
    os.chdir(project_root)

    from src.train import train as train_main

    print("[Modal] Starting trainign with Elo-filtered Lichess dataset...")
    train_main()
    print("[Modal] Training complete.")

@app.local_entrypoint()
def main():
    run_training.remote()
