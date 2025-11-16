from torch.utils.data import DataLoader
from data.local_pgn_dataset import LocalPGNDataset

def main():
    #  CHANGE THIS LINE ONLY 
    dataset = LocalPGNDataset("data/localGame.pgn", max_moves=40)
    #                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Replace with your actual PGN file name inside the data/ folder

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    x, policy_target, value_target = next(iter(loader))

    print("x shape:", x.shape)
    print("policy target:", policy_target)
    print("value target:", value_target)

if __name__ == "__main__":
    main()
