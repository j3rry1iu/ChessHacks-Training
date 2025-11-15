import chess
import numpy as np

def encode_board(board: chess.Board) -> np.ndarray:
    array = np.zeros((12,8,8), dtype=np.int8)
    
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        piece_type = piece.piece_type
        color = piece.color
        color_offset = 0 if color == chess.WHITE else 6
        idx = piece_type - 1

        array[ color_offset + idx, 7-rank, file] = 1
    return array

if __name__ == "__main__":
    board = chess.Board()
    print(board)
    encoded = encode_board(board)
    print(encoded)
