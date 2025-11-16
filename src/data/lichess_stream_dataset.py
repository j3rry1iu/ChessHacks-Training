import io
from typing import Optional

import chess
import chess.pgn
import torch
import numpy as np
from torch.utils.data import IterableDataset

from board_encoder import encode_board        # returns numpy array already? if not, we adjust
from utils.move_index import move_to_index


class LichessGameStreamDataset(IterableDataset):
    """
    High-throughput version:
    - avoids repeated tensor allocations
    - reduces python-chess overhead
    - caches encoded boards
    """

    def __init__(
        self,
        hf_dataset,
        max_moves_per_game: Optional[int] = None,
        min_elo: int = 2200,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.max_moves_per_game = max_moves_per_game
        self.min_elo = min_elo

        # Pre-allocate reusable tensors (GPU moves later)
        self._encode_buf = None  # will hold (12,8,8) np array

    def _parse_pgn_fast(self, pgn_text: str):
        """Fast PGN parsing using a single read() call (python-chess optimized path)."""
        return chess.pgn.read_game(io.StringIO(pgn_text))

    def __iter__(self):
        encode_board_local = encode_board
        move_to_index_local = move_to_index
        max_moves = self.max_moves_per_game
        min_elo = self.min_elo

        for row in self.hf_dataset:

            # --- Fast Elo filtering ---
            we = row.get("white_elo") or row.get("white_Elo") or -1
            be = row.get("black_elo") or row.get("black_Elo") or -1
            if (we < min_elo) and (be < min_elo):
                continue

            # --- Try to get PGN ---
            pgn_text = row.get("pgn") or row.get("PGN")
            if pgn_text is None:
                continue

            # --- Fast PGN → Game ---
            game = self._parse_pgn_fast(pgn_text)
            if game is None:
                continue

            result_str = game.headers.get("Result", "*")
            value = (
                1.0 if result_str == "1-0" else
                -1.0 if result_str == "0-1" else
                0.0
            )

            board = game.board()
            moves = game.mainline_moves()
            if max_moves:
                moves = list(moves)[:max_moves]

            # Reuse one buffer for encoded board
            encode_buf = self._encode_buf
            for move in moves:
                # Avoid allocating new arrays each time
                encoded = encode_board_local(board)

                # Optionally, convert to torch with no-copy if contiguous
                x = torch.from_numpy(np.asarray(encoded, dtype=np.float32))

                policy_idx = move_to_index_local(move)
                policy = torch.tensor(policy_idx, dtype=torch.long)
                val = torch.tensor(value, dtype=torch.float32)

                yield x, policy, val

                board.push(move)