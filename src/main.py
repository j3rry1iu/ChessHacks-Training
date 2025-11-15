import chess
import torch

from board_encoder import encode_board
from models.chess_net import ChessNet
from search.search import choose_best_move

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board = chess.Board()
    model = ChessNet().to(device)
    model.eval()

    best_move, best_score = choose_best_move(board, model, device, 3, 20, 10)
    print("Best move:", best_move, "Score:", best_score)