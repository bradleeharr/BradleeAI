import lichesspy.api
import chess
from pgn2bitboard import *
from lichesspy.format import PGN
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor


def main():
    NUM_GAMES = 3000
    USER = 'bubbakill7'
    pgn = lichesspy.api.user_games(USER, max=NUM_GAMES, format=PGN, evals=True)
    boards = np.zeros((10, 773, 773));
    idx = 0
    for game in pgn:
        pgn_name = 'pgn' + str(idx) + '.pgn'
        with open(pgn_name, 'w') as f:
            f.write(game)
        idx += 1


if __name__ == '__main__':
    main()

