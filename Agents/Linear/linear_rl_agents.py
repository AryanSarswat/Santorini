from Game import *
import numpy as np
import random
from linear_rl_core_v2 import Minimax, MinimaxWithPruning, LinearFnApproximatorV2
from fast_board import FastBoard

class SearchBootstrapper():
    def __init__(self, weights = None, learning_rate = 10**-5):
        self.NUM_WEIGHTS = 22
        self.learning_rate = learning_rate
        self.fast_board = FastBoard()
        if type(weights) == type(None):
            #randomly initialize weights btwn -1 and 1
            self.weights = np.array([random.uniform(-1,1) for i in range(self.NUM_WEIGHTS)])
        else:
            self.weights = np.array(weights)

    def __repr__(self):
        return f'{self.__class__.__name__} weights are {self.weights}'

    def update_weights(self, minimax_tree):
        pass

class RootStrapAB(SearchBootstrapper):
    '''
    performs pruned minimax search at current state, then updates parameters using SGD
    to approximated value of node closer to minimax search value
    '''

    def __init__(self, weights = None, learning_rate = 10**-5):
        super().__init__(weights, learning_rate)

    def update_weights(self, minimax_tree):
        #update weights of approximator towards minimax search value
        linear_approximator = LinearFnApproximatorV2(minimax_tree.board_levels, minimax_tree.all_worker_coords, self.weights, self.fast_board)
        approximated_value = linear_approximator.state_value
        feature_vector = linear_approximator.get_features()
        error = minimax_tree.value - approximated_value
        weight_update = self.learning_rate * error * feature_vector #feature vector is an np array
        self.weights += weight_update

class TreeStrapMinimax(SearchBootstrapper):
    '''
    performs full minimax search at current state, then for every single leaf node update
    parameters towards minimax search value using SGD
    '''
    def __init__(self, weights = None, learning_rate = 10**-6):
        super().__init__(weights, learning_rate)

    def update_weights(self, minimax_tree, root = True):
        total_weight_update = np.array([0.0 for i in range(self.NUM_WEIGHTS)])
        #if minimax_tree.depth == 1:
        total_weight_update += self.calculate_weight_update(minimax_tree)
        if minimax_tree.depth > 1:
            for child_node in minimax_tree.child_nodes:
                if child_node.winner == None:
                    total_weight_update += self.update_weights(child_node, False)
        if not root:
            return total_weight_update
        else:
            #print(total_weight_update)
            #only update self.weights at the end to 'freeze' state value approximation
            self.weights += total_weight_update

    def calculate_weight_update(self, node):
        linear_approximator = LinearFnApproximatorV2(node.board_levels, node.all_worker_coords, self.weights, self.fast_board)
        approximated_value = linear_approximator.state_value
        feature_vector = linear_approximator.get_features()
        error = node.value - approximated_value
        weight_update = self.learning_rate * error * feature_vector
        return weight_update
        
class RandomAgent():
    '''
    parent class for AI agent playing Santorini. by default, will play completely randomly.
    '''
    def __init__(self, name):
        self.name = name
        self.workers = [Worker([], str(name)+"1"), Worker([], str(name)+"2")]
        
    def place_workers(self, board):
        """
        Method to randomly place a player's worker on the board. in-place function.
        """
        workers_placed = 0
        while workers_placed < 2:
            try:
                coords = [random.randint(0, 5), random.randint(0, 5)]
                # Update own workers
                self.workers[workers_placed].update_location(coords)
                #updates directly into square of board (breaking abstraction barriers much?)
                board.board[coords[0]][coords[1]].update_worker(self.workers[workers_placed])
                workers_placed += 1
            except Exception:
                continue

        return board

    def action(self, board):
        """
        Method to select and place a worker, afterwards, place a building
        """
        board = random.choice(board.all_possible_next_states(self.name))
        return board

class LinearRlAgentV2(RandomAgent):
    '''
    basic RL agent using a linear function approximator and TD learning
    epsilon greedy policy too?

    Inputs: name: either 'A' or 'B', search depth, and trained_weights (when not in training mode)
    '''
    def __init__(self, name, search_depth, trained_weights = [0,2,4,0,-2,-4,-1,1], adaptive_search = False):
        super().__init__(name)
        self.search_depth = search_depth
        self.trained_weights = trained_weights
        self.adaptive_search = adaptive_search

    def action(self, board, trainer = None):
        """
        Method to select and place a worker, afterwards, place a building/
        If trainer is specified, will call corresponding search tree and update weights
        Otherwise, uses the specified weights and searches with a minimax tree with alpha beta pruning.
        """
        board_levels, all_worker_coords = FastBoard.convert_board_to_array(board)
        fast_board = FastBoard()
        if trainer != None:
            if isinstance(trainer, RootStrapAB):
                minimax_tree = MinimaxWithPruning(board_levels, all_worker_coords, self.name, self.search_depth, fast_board, trainer.weights, 'V1')
            elif isinstance(trainer, TreeStrapMinimax):
                minimax_tree = Minimax(board_levels, all_worker_coords, self.name, self.search_depth, fast_board, trainer.weights, 'V1')

            new_board_levels, new_worker_coords = minimax_tree.get_best_node()
            new_board = FastBoard.convert_array_to_board(board, new_board_levels, new_worker_coords)
            #update weights if in training mode.
            trainer.update_weights(minimax_tree)
        else:
            search_depth = self.search_depth
            #adaptive depth when not in training mode
            if self.adaptive_search:
                my_num_moves = len(fast_board.all_possible_next_states(board_levels, all_worker_coords, self.name))
                if self.name == 'A':
                    opponent = 'B'
                else: 
                    opponent = 'A'
                opp_num_moves = len(fast_board.all_possible_next_states(board_levels, all_worker_coords, opponent))
                if self.search_depth % 2 == 0:
                    next_search = self.name
                else:
                    next_search = opponent
                if my_num_moves + opp_num_moves < 20:
                    search_depth = self.search_depth + 3
                elif my_num_moves + opp_num_moves < 30:
                    search_depth = self.search_depth + 2
                elif my_num_moves + opp_num_moves < 40:
                    search_depth = self.search_depth + 1
                elif (my_num_moves < 20 and next_search == self.name) or (opp_num_moves < 20 and next_search == opponent):
                    search_depth = self.search_depth + 1
                print(f'Search Depth is {search_depth}, my moves = {my_num_moves}, opp moves = {opp_num_moves}')
            minimax_tree = MinimaxWithPruning(board_levels, all_worker_coords, self.name, search_depth, fast_board, self.trained_weights, 'V1')
            new_board_levels, new_worker_coords = minimax_tree.get_best_node()
            new_board = FastBoard.convert_array_to_board(board, new_board_levels, new_worker_coords)
        return new_board

class LinearRlAgentV3(RandomAgent):
    '''
    basic RL agent using a linear function approximator and TD learning
    epsilon greedy policy too?

    Inputs: name: either 'A' or 'B', search depth, and trained_weights (when not in training mode)
    '''
    def __init__(self, name, search_depth, trained_weights):
        super().__init__(name)
        self.search_depth = search_depth
        self.trained_weights = trained_weights

    def action(self, board, trainer = None):
        """
        Method to select and place a worker, afterwards, place a building/
        If trainer is specified, will call corresponding search tree and update weights
        Otherwise, uses the specified weights and searches with a minimax tree with alpha beta pruning.
        """
        board_levels, all_worker_coords = FastBoard.convert_board_to_array(board)
        fast_board = FastBoard()
        if trainer != None:
            if isinstance(trainer, RootStrapAB):
                minimax_tree = MinimaxWithPruning(board_levels, all_worker_coords, self.name, self.search_depth, fast_board, trainer.weights, 'V2')
            elif isinstance(trainer, TreeStrapMinimax):
                minimax_tree = Minimax(board_levels, all_worker_coords, self.name, self.search_depth, fast_board, trainer.weights, 'V2')

            new_board_levels, new_worker_coords = minimax_tree.get_best_node()
            new_board = FastBoard.convert_array_to_board(board, new_board_levels, new_worker_coords)
            #update weights if in training mode.
            trainer.update_weights(minimax_tree)
        else:
            minimax_tree = MinimaxWithPruning(board_levels, all_worker_coords, self.name, self.search_depth, fast_board, self.trained_weights, 'V2')
            new_board_levels, new_worker_coords = minimax_tree.get_best_node()
            new_board = FastBoard.convert_array_to_board(board, new_board_levels, new_worker_coords)
        return new_board