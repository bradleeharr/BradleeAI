import sys, chess, time
import chess.svg
from display_svg import *
from minmax import *
from chessboard import display

NODECOUNT = 0

def main():
    board = chess.Board("rnbqkbnr/1p1p1pp1/2p4p/p3p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 5")
    print("running")
    # board = chess.Board()
    # print(getNextMove(4, board, True))

    print('sleeping for 1s')
    time.sleep(1)
    print('done')
    b = chess.Board()
    game_board = display.start()
    display.update(board.fen(), game_board)

    print(getNextMove(2, board, True))
    while True:
        display.check_for_quit()
        display.update(board.fen(), game_board)
        white = (board.turn is chess.WHITE)
        next_move, _ = getNextMove(2, board, white)
        print(next_move)
        board.push(next_move)
        time.sleep(0.1);

    display.terminate()

    print(getNextMove(3, board, True))

    print(NODECOUNT)




if __name__ == '__main__':
    main()
