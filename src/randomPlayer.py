# -*- coding: utf-8 -*-
''' This is the famous random player which (almost) always looses.
'''

import Goban
from playerInterface import *

import numpy as np
import matplotlib.pyplot as plt

class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None

    def getPlayerName(self):
        return "Random Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        # Get the list of all possible moves
        moves = self._board.legal_moves() # Dont use weak_legal_moves() here!

        # Let's plot some board probabilities
        import go_plot
        # Generate random proabibilities
        probabilities = np.random.uniform(size=82)
        # Now we want to to put to 0 all impossible moves
        # SO we careate multiplier with 0 everywhere and put 1 where the move is legal
        multiplier = np.zeros_like(probabilities)
        for some_move in moves:
            x, y = Goban.Board.unflatten(some_move)
            multiplier[y * Goban.Board._BOARDSIZE + x] = 1
        # Pass move is always legal
        multiplier[-1] = 1
        # Now we multiply our probs
        probabilities *= multiplier
        # Normalize them
        probabilities /= np.sum(probabilities)

        # We plot them
        go_plot.plot_play_probabilities(self._board, probabilities)
        plt.show()


        move = np.random.choice(range(82), p=probabilities)
        # Correct number for PASS
        if move == 81:
            move = -1
        self._board.push(move)

        # New here: allows to consider internal representations of moves
        #print("I am playing ", self._board.move_to_str(move))
        #print("My current board :")
        #self._board.prettyPrint()

        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(move)

    def playOpponentMove(self, move):
        #print("Opponent played ", move, "i.e. ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
