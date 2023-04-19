import unittest
import chess
from from_colab import *
from from_colab.utilities import *


class BoardTest(unittest.TestCase):
    # Tests that the 2d input conversion board method will convert the position to the board
    def test_board_to_2d_input(self):
        board = chess.Board()
        input_board = board_to_input_2d(board, 0)
        print(board)
        print(input_board)
        reconverted_board = input_to_board_2d(input_board, 0)
        print(reconverted_board)
        self.assertEqual(reconverted_board, board)  # add assertion here

    def test_move_to_flat(self):
        board = chess.Board()
        from_square, to_square = ('e2', 'e4')
        move = chess.Move.from_uci(from_square + to_square)
        print(move)
        flat = move_to_flat(move)
        print(flat)
        unflat_move = flat_to_move(flat, board)
        print(unflat_move)
        self.assertEqual(unflat_move, move)

if __name__ == '__main__':
    unittest.main()
