import io
from typing import Optional

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset

from board_encoder import encode_board
from utils.move_index import move_to_index


def result_to_value(result_str: str) -> float:
    """
    Map PGN result string to scalar in [-1, 1] from White's perspective.
    """
    if not result_str:
        return 0.0
    result_str = result_str.strip()
    if result_str == "1-0":
        return 1.0
    if result_str == "0-1":
        return -1.0
    if result_str in ("1/2-1/2", "½-½"):
        return 0.0
    return 0.0


def _parse_int(value, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _extract_elo(row: dict, color: str) -> int:
    """
    Try multiple common rating field names for white/black.
    color: "white" or "black"
    """
    # Common naming variants in chess/lichess datasets
    keys = [
        f"{color}_elo",
        f"{color}_Elo",
        f"{color.capitalize()}Elo",
        f"{color}_rating",
        f"{color.capitalize()}Rating",
    ]

    for k in keys:
        if k in row and row[k] is not None:
            return _parse_int(row[k])

    # Fallback: some datasets might use generic "White" / "Black" subdicts
    # with an "elo" or "rating" field
    side = row.get(color) or row.get(color.capitalize())
    if isinstance(side, dict):
        for k in ("elo", "rating"):
            if k in side and side[k] is not None:
                return _parse_int(side[k])

    return -1


class LichessGameStreamDataset(IterableDataset):
    """
    Stream rows from a HuggingFace Lichess dataset and convert them into
    (board, policy_target, value_target) training samples.

    - hf_dataset: streaming HF dataset, e.g. load_dataset(..., streaming=True)["train"]
    - max_moves_per_game: limit how many moves you take from each game (for speed).
    - min_elo: minimum Elo rating threshold; if BOTH players are below this, the game is skipped.
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

    def _parse_pgn(self, pgn_text: str) -> Optional[chess.pgn.Game]:
        """Convert PGN string to python-chess Game object."""
        pgn_io = io.StringIO(pgn_text)
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception:
            return None
        return game

    def __iter__(self):
        """
        Yield training samples:
          x:           (12, 8, 8) float tensor
          policy_idx:  Long scalar = move index
          value:       Float scalar in [-1, 1]
        Only from games where at least one player has Elo >= min_elo.
        """
        for row in self.hf_dataset:
            # --- Elo filtering ---
            white_elo = _extract_elo(row, "white")
            black_elo = _extract_elo(row, "black")

            # Skip games where both players are below min_elo
            if white_elo < self.min_elo and black_elo < self.min_elo:
                continue

            # --- PGN / moves extraction ---
            pgn_text = row.get("pgn") or row.get("PGN") or None

            # Fallback: if only moves are provided, build a minimal PGN
            if pgn_text is None and "moves" in row:
                moves_str = row["moves"]
                # We use '*' result if unknown; value head will just see 0.0
                pgn_text = f"[Result \"*\"]\n\n{moves_str}"

            if pgn_text is None:
                continue

            game = self._parse_pgn(pgn_text)
            if game is None:
                continue

            result_str = game.headers.get("Result", "*")
            value_label = result_to_value(result_str)

            board = game.board()
            moves = list(game.mainline_moves())
            if self.max_moves_per_game is not None:
                moves = moves[: self.max_moves_per_game]

            for move in moves:
                encoded = encode_board(board)  # (12, 8, 8)
                policy_idx = move_to_index(move)

                x = torch.tensor(encoded, dtype=torch.float32)
                policy_target = torch.tensor(policy_idx, dtype=torch.long)
                value_target = torch.tensor(value_label, dtype=torch.float32)

                yield x, policy_target, value_target

                board.push(move)
