# src/smoke_test_model.py

import torch
from models.chess_net import ChessNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    model.eval()

    # Fake batch: (batch=2, channels=12, board=8x8)
    x = torch.randn(2, 12, 8, 8, device=device)

    with torch.no_grad():
        policy_logits, values = model(x)

    print("policy_logits shape:", policy_logits.shape)  # expect (2, NUM_MOVES)
    print("values shape:", values.shape)                # expect (2, 1)

if __name__ == "__main__":
    main()
