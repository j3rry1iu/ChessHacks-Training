# src/check_hf_slice.py
from datasets import load_dataset

def main():
    ds = load_dataset("Lichess/standard-chess-games", split="train[:10]")
    print("HF rows:", len(ds))
    for i in range(3):
        row = ds[i]
        print(f"Row {i} id={row.get('id')} result={row.get('result')}")
        print("  keys:", list(row.keys()))

if __name__ == "__main__":
    main()
