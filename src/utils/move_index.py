import chess

BASE_MOVE_SPACE = 64*64

PROMOTION_PIECES= [
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT
]

NUM_PROMOTIONS = len(PROMOTION_PIECES)

NUM_MOVES = BASE_MOVE_SPACE + BASE_MOVE_SPACE*NUM_PROMOTIONS

def move_to_index(move:chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    base = from_sq*64 + to_sq

    if move.promotion is None:
        return base
    try:
        promo_type_index = PROMOTION_PIECES.index(move.promotion)
    except ValueError:
        promo_type_index = 0
    
    return BASE_MOVE_SPACE + base* NUM_PROMOTIONS + promo_type_index

def index_to_move(idx:int) -> chess.Move:
    if idx < BASE_MOVE_SPACE:
        from_sq = idx // 64
        to_sq = idx%64
        return chess.Move(from_sq, to_sq)
    tmp = idx-BASE_MOVE_SPACE
    base = tmp // NUM_PROMOTIONS
    promo_type_index = tmp % NUM_PROMOTIONS

    from_sq = base // 64
    to_sq = base % 64
    promotion_piece_type = PROMOTION_PIECES[promo_type_index]

    return chess.Move(from_sq, to_sq, promotion=promotion_piece_type)