from datasets import load_dataset
from torch.utils.data import DataLoader

from data.lichess_small_dataset import LichessSmallDataset


def main():
    print("Loading HF slice (non-streaming)...")
    hf_train = load_dataset("Lichess/standard-chess-games", split="train[:200]")

    dataset = LichessSmallDataset(hf_train, max_moves_per_game=20, max_samples=1000)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print("Getting one batch...")
    x, policy_target, value_target = next(iter(loader))

    print("x shape:", x.shape)                      # (4, 12, 8, 8)
    print("policy_target shape:", policy_target.shape)
    print("value_target shape:", value_target.shape)
    print("policy_target[0]:", policy_target[0])
    print("value_target[0]:", value_target[0])


if __name__ == "__main__":
    main()
