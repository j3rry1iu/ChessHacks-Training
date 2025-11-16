"""
Test script to verify Elite Chess Games dataset format.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets import load_dataset
from data.elite_chess_dataset import EliteChessDataset

print("Loading Elite Chess Games dataset...")
hf_train = load_dataset("Q-bert/Elite-Chess-Games", streaming=True, split="train")

print("\n=== Testing with all elite games (Elo >= 2600) ===")
dataset = EliteChessDataset(
    hf_dataset=hf_train,
    max_moves_per_game=10,  # Just test first 10 moves
    min_avg_elo=2600,
    player_filter=None,
)

print("Fetching first 5 positions...")
for i, (x, policy, value) in enumerate(dataset):
    if i >= 5:
        break
    print(f"Position {i+1}: x.shape={x.shape}, policy={policy.item()}, value={value.item():.3f}")

print(f"\nProcessed {dataset.game_count} games, {dataset.position_count} positions")

print("\n=== Testing with Magnus Carlsen games only ===")
hf_train = load_dataset("Q-bert/Elite-Chess-Games", streaming=True, split="train")
dataset_magnus = EliteChessDataset(
    hf_dataset=hf_train,
    max_moves_per_game=20,
    min_avg_elo=2600,
    player_filter="Carlsen",
)

print("Fetching first 10 Magnus positions...")
for i, (x, policy, value) in enumerate(dataset_magnus):
    if i >= 10:
        break
    print(f"Magnus position {i+1}: x.shape={x.shape}, policy={policy.item()}, value={value.item():.3f}")

print(f"\nProcessed {dataset_magnus.game_count} Magnus games, {dataset_magnus.position_count} positions")
print("\n✓ Format test successful!")
