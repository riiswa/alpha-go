from multiprocessing import Pool
from playerInterface import *
from Goban import Board
from node import MCTSNode

import math
import copy
import time

class myPlayer(PlayerInterface):

    def __init__(self):
        self._board = Board()
        self._mycolor = None

    def getPlayerName(self):
        return "Mcts Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            return "PASS"
        move = self.select_move(self._board)
        self._board.play_move(move)
        return Board.flat_to_name(move) 

    def playOpponentMove(self, move):
        self._board.play_move(Board.name_to_flat(move)) 

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won :D")
        else:
            print("I lost :(")

    @staticmethod
    def select_move(board_org, max_time=1, temperature=1.2):
        start_time = time.time()
        root = MCTSNode(board_org.weak_legal_moves())
        # add nodes
        i=0
        pool = Pool()
        while(True):
            board = copy.deepcopy(board_org)
            node = root
            while (not node.can_add_child()) and (not board.is_game_over()):
                node = myPlayer.select_child(node, board, temperature)
                #board.push(node.move)
            if node.can_add_child() and not board.is_game_over():
                node = node.add_random_child(board)
                #board.push(node.move)
            winners = []
            results = []
            # use all cores of the processor
            for proc in range(pool._processes):
                results.append(pool.apply_async(myPlayer.simulate_random_game, [board]))
            for res in results:
                winners.append(res.get())
            while node is not None:
                for winner in winners:
                    node.record_win(winner)
                node = node.parent
            if (time.time() - start_time >= max_time):
                print()
                break
            i+=pool._processes
            print("Rounds %d (%f)" % (i,time.time()-start_time), end='\r')
        # debug
        scored_moves = [(child.winning_frac(board_org.next_player()), child.move, child.num_rollouts)
                        for child in root.children]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:5]:
            print('%s - %.3f (%d)' % (m, s, n))
        # pick best node
        best_move = -1
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(board_org.next_player())
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move

    @staticmethod
    def select_child(node, board, temperature):
        # upper confidence bound for trees (UCT) metric
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # loop over each child.
        for child in node.children:
            # calculate the UCT score.
            win_percentage = child.winning_frac(board.next_player())
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + temperature * exploration_factor
            # Check if this is the largest we've seen so far.
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        board.play_move(best_child.move)
        return best_child

    @staticmethod
    def simulate_random_game(board):
        def is_point_an_eye(board, coord):
            # We must control 3 out of 4 corners if the point is in the middle
            # of the board; on the edge we must control all corners.
            friendly_corners = 0
            off_board_corners = 0
            i_org = i = board._neighborsEntries[coord]
            while board._neighbors[i] != -1:
                n = board._board[board._neighbors[i]]
                if  n == board.next_player():
                    return False
                if (n != Board._EMPTY) or (n != board.next_player()):
                    friendly_corners += 1
                i += 1
            if i >= i_org+4:
                # Point is in the middle.
                return friendly_corners >= 3
            # Point is on the edge or corner.
            return (4-i_org-i) + friendly_corners == 4
        # ==============================
        while not board.is_game_over():
            moves = board.weak_legal_moves()
            random.shuffle(moves)
            valid_move = -1 # PASS
            for move in moves:
                if not(is_point_an_eye(board, move)) and (board.play_move(move)):
                    valid_move = move
                    break
            if valid_move == -1:
                board.play_move(-1)
        if (board._nbWHITE > board._nbBLACK):
            return "1-0"
        elif (board._nbWHITE < board._nbBLACK):
            return "0-1"
        else:
            return "1/2-1/2"
        #return board.result()
