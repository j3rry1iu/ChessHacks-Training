# modal_train.py

import modal

# Name your Modal app
app = modal.App("chesshacks-train")

# Define your image: base Python + needed pip deps
image = (
    modal.Image.debian_slim()
    .apt_install("git")  # optional
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "python-chess",
        "datasets",
        "numpy",
    )
)

# Volume to persist weights between runs
weights_vol = modal.Volume.from_name("chesshacks-weights", create_if_missing=True)


@app.function(
    image=image,
    # Request GPU (A10G is a good mid-range choice; you can change type)
    gpu="A10G",
    timeout=60 * 60 * 3,  # 3 hours in seconds
    volumes={"/root/ChessHacks-Training/weights": weights_vol},
)
def run_training():
    """
    This runs inside Modal in the container.
    It assumes your repo is synced into /root/ChessHacks-Training.
    """
    import os
    import sys

    # Ensure project root is on sys.path
    project_root = "/root/ChessHacks-Training"
    sys.path.append(project_root)

    # Change working directory
    os.chdir(project_root)

    from src.train import train

    # Call your training function with whatever hyperparams you want in the cloud
    train(
        epochs=5,             # tune based on time
        steps_per_epoch=5000, # more steps = more training
        batch_size=256,
        max_moves_per_game=80,
        lr=1e-3,
    )


@app.local_entrypoint()
def main():
    """
    Entry point when you run: modal run modal_train.py
    """
    run_training.remote()


@app.function(
    image=image,
    volumes={"/root/ChessHacks-Training/weights": weights_vol},
)
def list_weights():
    import os
    base = "/root/ChessHacks-Training/weights"
    print("Files in weights/:", os.listdir(base))


@app.function(
    image=image,
    volumes={"/root/ChessHacks-Training/weights": weights_vol},
)
def get_weight_file(name: str) -> bytes:
    """
    Return raw bytes of a weight file so we can save it locally.
    """
    import os
    base = "/root/ChessHacks-Training/weights"
    path = os.path.join(base, name)
    with open(path, "rb") as f:
        return f.read()
