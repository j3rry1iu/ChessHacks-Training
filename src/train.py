import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import chess.pgn
import io

def loadData():

    dataset = load_dataset("lichess_games", streaming=True)

    for d in dataset['train']:
        pgn = io.StringIO(d['text'])
        game = chess.pgn.read_game(pgn)
        