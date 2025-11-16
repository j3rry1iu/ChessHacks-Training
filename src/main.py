import chess
import torch

from board_encoder import encode_board
from models.chess_net import ChessNet
from search.search import choose_best_move
from datasets import load_dataset
import chess.pgn
import io


min_elo = 2200



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board = chess.Board()
    model = ChessNet().to(device)
    model.eval()
    
    dataset = load_dataset("Lichess/standard-chess-games", streaming=True)['train']
    print(f"Streaming dataset... filtering for games >= {min_elo} ELO.")
    count = 0
    for data in dataset:
        if count > 1:
            break
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
            count += 1
            board = game.board()        
            best_move, best_score = choose_best_move(board, model, device, 3, 20, 10)
            

        except Exception as e:
                # Skip bad games (e.g., parsing errors, ELO not a number)
                # print(f"Skipping game due to error: {e}")
                continue
            