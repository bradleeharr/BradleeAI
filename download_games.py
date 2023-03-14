import lichesspy.api
import chess
from pgn2bitboard import *
from lichesspy.format import PGN
import numpy as np


def pgn_to_bitboard(pgn_file_path, user):
    """
    Iterate over all games of pgnFile and returns some good positions from those games
    Modified to add extra outputs

    Inputs:
        pgnFilePath: path of the file where pgn games are stored
    Outputs:
        bitboards: chessboard positions [np array of shape(None, 773)]
        labels:
            Index 0: winner [np array of shape(None, 1)]
            Index 1: if user ("bradlee") is W or B
            Index 2: The next move made by user ("bradlee")

    """
    pgnFile = open(pgn_file_path)
    game = chess.pgn.read_game(pgnFile)
    bitboards = np.ndarray((0, 773))
    labels = np.ndarray((0, 1))
    count = 0

    while game is not None:
        win = winner(game)
        # player = get_player(game)
        if win in ['w', 'b']:
            positions, moves = pgn2fen(game)
            positions = choosePositions(positions, moves)
            for position in positions:
                bitboard = fen2bitboard(position)
                label = 1 if win == 'w' else 0
                # label[1] = 1 if player == user else 0
                # label[2] = nextMove(game)
                label = np.array([[label]])
                bitboards = np.append(bitboards, bitboard, axis=0)
                labels = np.append(labels, label, axis=0)
                count += 1

        game = chess.pgn.read_game(pgnFile)
    return bitboards, labels


def main():
    NUM_GAMES = 3
    USER = 'bubbakill7'
    pgn = lichesspy.api.user_games(USER, max=NUM_GAMES, format=PGN, evals=True)
    boards = np.zeros((10, 773, 773));
    idx = 0
    for game in pgn:
        pgn_name = 'pgn' + str(idx) + '.pgn'
        with open(pgn_name, 'w') as f:
            f.write(game)
        bitboards, labels = pgn_to_bitboard(pgn_name, USER)
        print(str(bitboards) + str(idx))
        idx += 1



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
