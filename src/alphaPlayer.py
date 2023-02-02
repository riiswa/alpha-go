from multiprocess import Pool

from playerInterface import *
from Goban import Board
from node import MCTSNode
import math
import copy
import time

from net import *
from feature_extraction import SGFDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


color_dict = {Board._BLACK: 'b', Board._WHITE: 'w'}

class myPlayer(PlayerInterface):

    def __init__(self):
        self._board = Board()
        self._mycolor = None

        feature_network1 = FeatureExtractor(11, 128, 6)
        self.policy_network = GoNetwork(
            feature_network1,
            81,
            nn.functional.softmax,
            lambda x, x_: x * x_[:, -1, :, :].flatten(start_dim=1)
        )
        self.policy_network.load_state_dict(torch.load("weights/policy_weights.pt", map_location=torch.device('cpu')))

        feature_network2 = FeatureExtractor(11, 128, 6)
        self.value_network = GoNetwork(feature_network2, 1, nn.functional.sigmoid)
        self.value_network.load_state_dict(torch.load("weights/value_weights.pt", map_location=torch.device('cpu')))

    def getPlayerName(self):
        return "Alpha Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            return "PASS"
        # try:
        move = self.select_move(self._board, self._mycolor, self.value_network, self.policy_network)
        # except RuntimeError as e:
        #     print("e")
        #     exit()
        #     return "PASS"

        self._board.push(move)
        return Board.flat_to_name(move)

    def playOpponentMove(self, move):
        self._board.push(Board.name_to_flat(move))

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")

    @staticmethod
    def select_move(board_org, color, value_net, policy_net, max_time=2, cpuct=1.41):
        start_time = time.time()
        root = MCTSNode(board_org.weak_legal_moves())
        # add nodes
        i = 0
        nb_rollout = 0
        pool = Pool()
        while (True):
            board = copy.deepcopy(board_org)
            node = root
            while (not node.can_add_child()) and (not board.is_game_over()):
                node = myPlayer.select_child(node, board, cpuct)

            if node.can_add_child() and not board.is_game_over():
                node = node.add_random_child(board)

            results = []
            for _ in range(pool._processes):
                results.append(pool.apply_async(myPlayer.simulate_random_game, [board, policy_net]))
            values = []
            for res in results:
                values.append(
                    value_net(SGFDataset.get_feature_from_board(board, color_dict[board.next_player()]).reshape((11, 9, 9)).unsqueeze(0)) \
                        .squeeze().item()
                )

            while node is not None:
                for value in values:
                    node.record_value(value)
                node = node.parent

            if (time.time() - start_time >= max_time):
                print("Number of rollouts", nb_rollout)
                break

            i += pool._processes
            nb_rollout += 1

        # pick best node
        best_move = -1
        best_pct = -1.0
        for child in root.children:
            child_pct = child.value_frac()
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        print('Best move %s with pct %.3f' % (best_move, best_pct))
        return best_move

    @staticmethod
    def uct(node, child, board, cpuct):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        win_percentage = child.value_frac()
        exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
        return win_percentage + cpuct * exploration_factor

    @staticmethod
    def select_child(node, board, cpuct):
        best_score = -1
        best_child = None
        for child in node.children:
            uct_score = myPlayer.uct(node, child, board, cpuct)
            # choose best UCT
            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        board.play_move(best_child.move)
        return best_child

    @staticmethod
    def simulate_random_game(board, policy_net):
        def is_point_an_eye(board, coord):
            friendly_corners = 0
            i_org = i = board._neighborsEntries[coord]
            while board._neighbors[i] != -1:
                n = board._board[board._neighbors[i]]
                if n == board.next_player():
                    return False

                if (n != Board._EMPTY) or (n != board.next_player()):
                    friendly_corners += 1
                i += 1

            if i >= i_org + 4:
                return friendly_corners >= 3
            return (4 - i_org - i) + friendly_corners == 4

        while not board.is_game_over():
            moves = policy_net(SGFDataset.get_feature_from_board(board, color_dict[board.next_player()]).reshape((11, 9, 9)).unsqueeze(0))\
                .squeeze().argsort(descending=True)
            valid_move = -1  # PASS

            for move in moves:
                if not (is_point_an_eye(board, move)) and (board.play_move(move)):
                    valid_move = move
                    break

            if valid_move == -1:
                board.play_move(-1)
        return board
