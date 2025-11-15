import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import chess.pgn
import io

def loadData(min_elo=2200):
    """
    This is a generator function that streams data from Lichess,
    filters it by ELO, and yields training tuples.
    """
    # Note: Using 'Icannos/lichess_games' as it seems more robust
    dataset = load_dataset("Icannos/lichess_games", streaming=True)['train']
    print(f"Streaming dataset... filtering for games >= {min_elo} ELO.")

    for data in dataset:
        try:
            # The 'pgn' field is correct for 'Icannos/lichess_games'
            pgn = io.StringIO(data['pgn']) 
            game = chess.pgn.read_game(pgn)
            
            if not game:
                continue

            # --- ELO Filter ---
            white_elo = int(game.headers.get('WhiteElo', 0))
            black_elo = int(game.headers.get('BlackElo', 0))

            if white_elo < min_elo or black_elo < min_elo:
                continue  # Skip low-rated game
            
            # --- Process Valid Game ---
            board = game.board()


        except Exception as e:
            # Skip bad games (e.g., parsing errors, ELO not a number)
            # print(f"Skipping game due to error: {e}")
            continue