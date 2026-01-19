# ChessHacks-Training
MyPyTorchBot - Lichess Chess Bot
A custom chess engine integrated with lichess-bot, powered by a PyTorch neural network (ChessNet) and hosted weights on Hugging Face.

🚀 Features
Automatic Weight Management: Automatically pulls the latest model weights (best.pt) from Hugging Face on startup.

PyTorch Backend: Uses a custom ChessNet architecture for move evaluation.

Lichess Integration: Fully compatible with the lichess-bot framework using the MinimalEngine wrapper.

Smart Caching: Weights are cached locally to ensure fast subsequent startups.
