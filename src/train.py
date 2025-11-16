import time
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset

from models.chess_net import ChessNet
from data.lichess_stream_dataset import LichessGameStreamDataset


WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, global_step, avg_loss, name: str):
    ckpt_path = WEIGHTS_DIR / name
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_loss": avg_loss,
        },
        ckpt_path,
    )
    print(f"[Checkpoint] Saved {name} (epoch={epoch}, step={global_step}, loss={avg_loss:.4f})")


def train(
    epochs: int = 3,
    steps_per_epoch: int = 2000,
    batch_size: int = 256,
    max_moves_per_game: int | None = 80,
    lr: float = 1e-3,
    min_elo: int = 2200,
):
    """
    Train ChessNet on Elo-filtered Lichess games using a streaming HF dataset.

    - epochs:           number of epochs (logical passes of steps_per_epoch steps)
    - steps_per_epoch:  how many gradient steps per epoch (since dataset is streaming)
    - batch_size:       batch size for DataLoader
    - max_moves_per_game: cap moves taken from each game (for speed)
    - lr:               learning rate
    - min_elo:          minimum Elo; games where BOTH players are < min_elo are skipped
    """

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # 1) Load streaming HF dataset
    print("[Train] Loading Lichess streaming dataset...")
    hf_train = load_dataset("Lichess/standard-chess-games", streaming=True)["train"]

    # Optional: shuffle with buffer for more variety
    hf_train = hf_train.shuffle(seed=42, buffer_size=10_000)

    # 2) Wrap in Elo-filtered streaming IterableDataset
    dataset = LichessGameStreamDataset(
        hf_dataset=hf_train,
        max_moves_per_game=max_moves_per_game,
        min_elo=min_elo,
    )

    # 3) DataLoader over streaming dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,       # keep 0 for simplicity / easier debugging
    )

    # 4) Model + optimizer
    model = ChessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        model.train()

        epoch_loss = 0.0
        steps_in_epoch = 0

        data_iter = iter(dataloader)
        start_time = time.time()

        for step in range(steps_per_epoch):
            try:
                x, policy_target, value_target = next(data_iter)
            except StopIteration:
                # Recreate iterator if we exhaust the stream (unlikely with streaming)
                data_iter = iter(dataloader)
                x, policy_target, value_target = next(data_iter)

            x = x.to(device)                         # (B, 12, 8, 8)
            policy_target = policy_target.to(device) # (B,)
            value_target = value_target.to(device)   # (B,)

            policy_logits, values = model(x)         # (B, NUM_MOVES), (B, 1)
            values = values.squeeze(-1)              # (B,)

            policy_loss = F.cross_entropy(policy_logits, policy_target)
            value_loss = F.mse_loss(values, value_target)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            steps_in_epoch += 1
            epoch_loss += loss.item()

            if global_step % 100 == 0:
                print(f"[Step {global_step}] loss={loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(1, steps_in_epoch)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f} time={elapsed:.1f}s")

        # Always save "last" checkpoint
        save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, "last.pt")

        # Save "best" if loss improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, "best.pt")

        # Optional: per-epoch snapshot
        # save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, f"epoch_{epoch}.pt")

    # Save plain state_dict for inference
    state_dict_path = WEIGHTS_DIR / "chess_net_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    print(f"[Train] Saved final state_dict to {state_dict_path}")
    print(f"[Train] Best avg epoch loss={best_loss:.4f}")


def main():
    # Defaults are reasonable for a first Modal run; you can tune them
    train(
        epochs=3,
        steps_per_epoch=3000,
        batch_size=256,
        max_moves_per_game=80,
        lr=1e-3,
        min_elo=2200, 
    )


if __name__ == "__main__":
    main()
