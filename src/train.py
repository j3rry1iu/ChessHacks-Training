import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import chess.pgn
import io


dataset = load_dataset("lichess_games", streaming=True)

for d in dataset['train']:
    pgn = io.StringIO(d['text'])
    game = chess.pgn.read_game(pgn)
    print(game.headers['White'], game.headers['Black'])
    print(game.headers['Result'])
    print(game.mainline_moves())
    break

