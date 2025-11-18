import io
import chess
import chess.pgn
import torch
import numpy as np
from torch.utils.data import IterableDataset
from board_encoder import encode_board
from utils.move_index import move_to_index

class LichessGameStreamDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset,
        max_moves_per_game: int | None = None,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.max_moves_per_game = max_moves_per_game

    def _parse_pgn_fast(self, pgn_text: str):
        return chess.pgn.read_game(io.StringIO(pgn_text))

    def __iter__(self):
        encode_board_local = encode_board
        move_to_index_local = move_to_index
        max_moves = self.max_moves_per_game

        games_processed = 0
        positions_yielded = 0

        for row in self.hf_dataset:
            games_processed += 1
            
            # --- LOGGING ---
            if games_processed % 1000 == 0:
                print(f"[Dataset] Processed {games_processed} games, yielded {positions_yielded} positions")

            # ==============================================================
            # FORMAT 1: "angeluriot" (List of moves + 'winner' column)
            # ==============================================================
            if "moves_san" in row:
                # 1. Get Moves (List of strings)
                moves_list = row.get("moves_san")
                if not moves_list: 
                    continue
                
                # 2. Get Result
                winner = row.get("winner") # 'white', 'black', or 'draw'
                value = 0.0
                if winner == "white": value = 1.0
                elif winner == "black": value = -1.0

                # 3. Replay Game
                board = chess.Board()
                for i, move_san in enumerate(moves_list):
                    if max_moves and i >= max_moves: break
                    
                    try:
                        move = board.parse_san(move_san)
                    except ValueError:
                        break # Invalid move in dataset, stop game

                    # Yield BEFORE pushing (predict the move about to be made)
                    encoded = encode_board_local(board)
                    yield (
                        torch.from_numpy(np.asarray(encoded, dtype=np.float32)),
                        torch.tensor(move_to_index_local(move), dtype=torch.long),
                        torch.tensor(value, dtype=torch.float32)
                    )
                    positions_yielded += 1
                    board.push(move)
                
                continue # Done with this game, skip the PGN logic below

            # ==============================================================
            # FORMAT 2: Standard Lichess (PGN string column)
            # ==============================================================
            pgn_text = row.get("pgn") or row.get("PGN") or row.get("pgn_full")
            
            if not pgn_text:
                movetext = row.get("movetext") or row.get("Moves") or row.get("moves")
                if not movetext:
                    continue
                pgn_text = f'{movetext}'

            game = self._parse_pgn_fast(pgn_text)
            if game is None:
                continue

            # Extract Result
            result_str = row.get("Result") or row.get("result") or game.headers.get("Result", "*")
            value = 0.0
            if result_str == "1-0": value = 1.0
            elif result_str == "0-1": value = -1.0

            board = game.board()
            moves = list(game.mainline_moves())
            
            if max_moves:
                moves = moves[:max_moves]

            for move in moves:
                encoded = encode_board_local(board)
                yield (
                    torch.from_numpy(np.asarray(encoded, dtype=np.float32)),
                    torch.tensor(move_to_index_local(move), dtype=torch.long),
                    torch.tensor(value, dtype=torch.float32)
                )
                positions_yielded += 1
                board.push(move)