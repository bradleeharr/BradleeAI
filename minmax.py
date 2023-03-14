import sys, chess, time
import chess.svg
from display_svg import *


# P = 100
# N = 310
# B = 320
# R = 500
# Q = 900

# position table
pieceSquareTable = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]]


def eval(board):
    scoreWhite = 0
    scoreBlack = 0
    for i in range(0, 8):
        for j in range(0, 8):
            squareIJ = chess.square(i, j)
            pieceIJ = board.piece_at(squareIJ)
            if str(pieceIJ) == "P":
                scoreWhite += (100 + pieceSquareTable[i][j])
            if str(pieceIJ) == "N":
                scoreWhite += (310 + pieceSquareTable[i][j])
            if str(pieceIJ) == "B":
                scoreWhite += (320 + pieceSquareTable[i][j])
            if str(pieceIJ) == "R":
                scoreWhite += (500 + pieceSquareTable[i][j])
            if str(pieceIJ) == "Q":
                scoreWhite += (900 + pieceSquareTable[i][j])
            if str(pieceIJ) == "p":
                scoreBlack += (100 + pieceSquareTable[i][j])
            if str(pieceIJ) == "n":
                scoreBlack += (310 + pieceSquareTable[i][j])
            if str(pieceIJ) == "b":
                scoreBlack += (320 + pieceSquareTable[i][j])
            if str(pieceIJ) == "r":
                scoreBlack += (500 + pieceSquareTable[i][j])
            if str(pieceIJ) == "q":
                scoreBlack += (900 + pieceSquareTable[i][j])
    return scoreWhite - scoreBlack

NODECOUNT = 0
def minimax(board, depth, maximize):
    global NODECOUNT
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -100000
        else:
            return 1000000
    if (board.is_stalemate()) or board.is_insufficient_material():
        return 0
    if depth == 0:
        return eval(board)
    legals = board.legal_moves
    if maximize:
        bestVal = -9999
        for move in legals:
            board.push(move)
            NODECOUNT += 1
            bestVal = max(bestVal, minimax(board, depth - 1, (not maximize)))
            board.pop()
        return bestVal
    else:
        bestVal = 9999
        for move in legals:
            board.push(move)
            NODECOUNT += 1
            bestVal = min(bestVal, minimax(board, depth - 1, (not maximize)))
            board.pop()
        return bestVal


def getNextMove(depth, board, maximize):
    legals = board.legal_moves
    bestMove = None
    bestValue = -9999
    if not maximize:
        bestValue = 9999
    for move in legals:
        board.push(move)
        value = minimax(board, depth - 1, (not maximize))
        board.pop()
        if maximize:
            if value > bestValue:
                bestValue = value
                bestMove = move
        else:
            if value < bestValue:
                bestValue = value
                bestMove = move
    return bestMove, bestValue
