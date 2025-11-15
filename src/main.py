from board_encoder import encode_board
from models.chess_net import ChessNet
import chess
import torch

if __name__ == "__main__":
    board = chess.Board()
    model = ChessNet()
    model.eval()  # Set to evaluation mode
    
    best_move = None
    best_value = float('-inf')  # Initialize variables
    
    for move in board.legal_moves:
        # Try the move
        board.push(move)
        
        # Evaluate resulting position
        encoded = encode_board(board)  # Shape: (12, 8, 8)
        
        # Convert to torch tensor with correct dtype and add batch dimension
        encoded_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        # Now shape is: (1, 12, 8, 8) - batch_size=1
        
        with torch.no_grad():  # No gradients needed for inference
            value = model(encoded_tensor)  # Returns tensor with shape (1, 1)
            value = value.item()  # Extract scalar value
        
        # Fix the logic - we're evaluating from current player's perspective
        # After board.push(), it's opponent's turn
        if board.turn == chess.BLACK:  # White just moved
            move_value = value  # Positive is good for white
        else:  # Black just moved
            move_value = -value  # Negative is good for black
            
        if move_value > best_value:
            best_value = move_value
            best_move = move
        
        # Undo the move
        board.pop()
    print(f"Best move: {best_move}, Value: {best_value}")