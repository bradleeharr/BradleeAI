import lichesspy.api
from lichesspy.format import PGN
import numpy as np
import chess
import chess.pgn


# Download lichess games for a particular user to a pgn file
def download_games_to_pgn(player_username, max_games=15):
    pgn = lichesspy.api.user_games(player_username, max=max_games, format=PGN, evals=True)
    idx = 0
    for game in pgn:
        with open(player_username + '_lichess_games.pgn', 'a') as f:
            f.write(game)
        idx += 1
    print(str(idx) + " Games downloaded")
    return idx


# Loads the games from a pgn file and places them into an array
def load_games_from_pgn(pgn_file):
    games = []
    with open(pgn_file) as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                # removes empty and  non-standard games
                # (Variants like Antichess or Atomic Chess)
                if game is None or game.headers["Variant"] != 'Standard':
                    break
                games.append(game)
            except Exception as e:
                print(f"Error reading PGN: {e}")
                break
    print(str(len(games)) + ' Games loaded.')
    return games


# Converts a board to a 3D array suitable for the CNN input
def board_to_input(board, num_previous_positions):
    pieces_order = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
    board_input = np.zeros((12 * (num_previous_positions + 1), 8, 8), dtype=np.float32)

    # Fill initial position
    for i, piece in enumerate(pieces_order):
        for square in board.pieces(piece, chess.WHITE):
            board_input[i, chess.square_rank(square), chess.square_file(square)] = 1
        for square in board.pieces(piece, chess.BLACK):
            board_input[i + 6, chess.square_rank(square), chess.square_file(square)] = 1

    # Fill previous positions
    for k in range(num_previous_positions):
        if len(board.move_stack) > 0:
            last_move = board.peek()
            board.pop()
        else:
            break

        for i, piece in enumerate(pieces_order):
            for square in board.pieces(piece, chess.WHITE):
                board_input[12 * (k + 1) + i, chess.square_rank(square), chess.square_file(square)] = 1
            for square in board.pieces(piece, chess.BLACK):
                board_input[12 * (k + 1) + i + 6, chess.square_rank(square), chess.square_file(square)] = 1
    return board_input


def input_to_board(board_input, num_previous_positions):
    pieces_order = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
    board = chess.Board()
    board.clear_board()

    for i, piece in enumerate(pieces_order):
        for rank in range(8):
            for file in range(8):
                if board_input[i, rank, file] == 1:
                    square = chess.square(file, rank)
                    board.set_piece_at(square, chess.Piece(piece, chess.WHITE))
                if board_input[i + 6, rank, file] == 1:
                    square = chess.square(file, rank)
                    board.set_piece_at(square, chess.Piece(piece, chess.BLACK))

    return board


# Processes a game to get all target player positions and move results
def process_game(game, target_player, num_previous_positions):
    board = game.board()
    positions = []
    target_moves = []
    player_to_move = game.headers["White"]

    for move_uci in game.mainline():
        move = chess.Move.from_uci(move_uci.uci())

        if board.turn == chess.WHITE:
            player_to_move = game.headers["White"]
        else:
            player_to_move = game.headers["Black"]

        if player_to_move == target_player:
            try:
                board_input = board_to_input(board.copy(), num_previous_positions)
                positions.append(board_input)
                target_moves.append(move_to_flat(move))
                board.push(move)
            except AssertionError:
                print(f"Skipping move {move} due to an error")
        else:
            try:
                board.push(move)
            except AssertionError:
                print(f"Skipping move {move} due to an error")

    return positions, target_moves


# Converts a move from square and to square to a flat number
def move_to_flat(move):
    from_square = move.from_square
    to_square = move.to_square
    # Consider all pawn promotions the same... Too rare to promote to non-queen
    # promotion = 1 if move.promotion else 0
    # Actually don't consider promotion at all, if a pawn moves to 8th rank it will promote regardless
    return from_square * 64 + to_square


# Converts the flat number to a move from square and to square
def flat_to_move(flat_move, board):
    from_square = flat_move // 64
    to_square = flat_move % 64
    move = chess.Move(from_square, to_square)

    # Promotion handling: Assume pawn promotion to queen
    from_rank, to_rank = chess.square_rank(from_square), chess.square_rank(to_square)
    if (to_rank == 0 or to_rank == 7) and board.piece_at(from_square).piece_type == chess.PAWN:
        move.promotion = chess.QUEEN

    return move
