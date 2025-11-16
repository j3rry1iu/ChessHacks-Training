# src/data/lichess_stream_dataset.py

import chess
import chess.pgn
import io
import torch
from torch.utils.data import IterableDataset

from board_encoder import encode_board
from utils.move_index import move_to_index


def result_to_value(result_str: str) -> float:
    if result_str == "1-0":
        return 1.0
    if result_str == "0-1":
        return -1.0
    if result_str in ("1/2-1/2", "½-½"):
        return 0.0
    return 0.0


class LichessGameStreamDataset(IterableDataset):
    """
    Stream PGN rows from HuggingFace dataset and convert to training samples.
    Suitable for extremely large datasets + Modal GPUs.
    """

    def __init__(self, hf_dataset, max_moves_per_game=None):
        self.hf_dataset = hf_dataset
        self.max_moves_per_game = max_moves_per_game

    def parse_pgn(self, pgn_text: str):
        """Convert PGN string → python-chess Game object"""
        pgn_io = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
        return game

    def __iter__(self):
        for row in self.hf_dataset:
            # Skip if PGN missing
            if "pgn" not in row:
                continue

            game = self.parse_pgn(row["pgn"])
            if game is None:
                continue

            value_label = result_to_value(game.headers.get("Result", "*"))

            board = game.board()
            moves = list(game.mainline_moves())

            if self.max_moves_per_game:
                moves = moves[:self.max_moves_per_game]

            for move in moves:
                encoded = encode_board(board)  # (12, 8, 8)
                move_idx = move_to_index(move)

                yield (
                    torch.tensor(encoded, dtype=torch.float32),
                    torch.tensor(move_idx, dtype=torch.long),
                    torch.tensor(value_label, dtype=torch.float32),
                )

                board.push(move)
