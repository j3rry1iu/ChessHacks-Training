# src/smoke_test_dataset.py

from datasets import load_dataset
from torch.utils.data import DataLoader

from data.lichess_stream_dataset import LichessGameStreamDataset


def main():
    print("Loading Lichess HF dataset (streaming)...")
    hf_train = load_dataset("Lichess/standard-chess-games", streaming=True)["train"]

    # Optional: shuffle with small buffer
    hf_train = hf_train.shuffle(seed=42, buffer_size=1000)

    dataset = LichessGameStreamDataset(hf_train, max_moves_per_game=20)

    # IMPORTANT: num_workers=0 so errors surface clearly
    loader = DataLoader(dataset, batch_size=4, num_workers=0)

    print("Created DataLoader, getting iterator...")
    it = iter(loader)

    print("Calling next(it)...")
    x, policy_target, value_target = next(it)
    print("Got a batch!")

    print("x shape:", x.shape)                      # expect (4, 12, 8, 8)
    print("policy_target shape:", policy_target.shape)  # expect (4,)
    print("value_target shape:", value_target.shape)    # expect (4,)
    print("policy_target example:", policy_target[0])
    print("value_target example:", value_target[0])


if __name__ == "__main__":
    main()
