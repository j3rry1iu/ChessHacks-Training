# src/smoke_test_dataset_raw.py

from datasets import load_dataset
from data.lichess_stream_dataset import LichessGameStreamDataset

def main():
    print("Loading Lichess HF dataset (streaming)...")
    hf_train = load_dataset("Lichess/standard-chess-games", streaming=True)["train"]

    # ⚠️ For debugging: do NOT shuffle here yet
    # hf_train = hf_train.shuffle(seed=42, buffer_size=1000)

    dataset = LichessGameStreamDataset(hf_train, max_moves_per_game=10)

    print("Iterating over raw dataset...")
    count = 0
    for x, policy_target, value_target in dataset:
        print("Sample", count)
        print("  x shape:", x.shape)
        print("  policy_target:", policy_target)
        print("  value_target:", value_target)
        count += 1
        if count >= 3:
            break

    print("Done. Total samples seen:", count)

if __name__ == "__main__":
    main()
