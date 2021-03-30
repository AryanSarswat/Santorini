from Game import *
import numpy as np
import random
from linear_rl_core_v2 import Minimax, MinimaxWithPruning, LinearFnApproximator
from fast_board import FastBoard
from linear_rl_launcher import run_santorini

class SearchBootstrapper():
    def __init__(self, weights = None, learning_rate = 10**-5):
        self.NUM_WEIGHTS = 8
        self.learning_rate = learning_rate
        if weights == None:
            #randomly initialize weights btwn -1 and 1
            self.weights = np.array([random.uniform(-1,1) for i in range(self.NUM_WEIGHTS)])
        else:
            self.weights = np.array(weights)

    def __repr__(self):
        return f'Current weights are {self.weights}'

    def update_weights(self, minimax_tree, board_levels, all_worker_coords):
        pass

class RootStrapAB(SearchBootstrapper):
    '''
    performs pruned minimax search at current state, then updates parameters using SGD
    to approximated value of node closer to minimax search value
    '''

    def __init__(self, weights = None, learning_rate = 10**-5):
        super().__init__(weights, learning_rate)

    def update_weights(self, minimax_tree, board_levels, all_worker_coords):
        #update weights of approximator towards minimax search value
        linear_approximator = LinearFnApproximator(board_levels, all_worker_coords, self.weights)
        approximated_value = linear_approximator.state_value
        feature_vector = linear_approximator.get_features()
        error = minimax_tree.value - approximated_value
        weight_update = self.learning_rate * error * feature_vector #feature vector is an np array
        self.weights += weight_update

class TreeStrapMinimax(SearchBootstrapper):
    '''
    performs full minimax search at current state, then for each leaf node update
    parameters towards minimax search value using SGD
    '''
    def __init__(self, weights = None, learning_rate = 10**-6):
        super().__init__(weights, learning_rate)

    def update_weights(self, minimax_tree, board_levels, all_worker_coords):
        total_weight_update = np.array([0.0 for i in range(self.NUM_WEIGHTS)])
        for child_node in minimax_tree.child_nodes:
            linear_approximator = LinearFnApproximator(child_node.board_levels, child_node.all_worker_coords, self.weights)
            approximated_value = linear_approximator.state_value
            feature_vector = linear_approximator.get_features()
            error = child_node.value - approximated_value
            weight_update = self.learning_rate * error * feature_vector
            total_weight_update += weight_update
        #only update self.weights at the end to 'freeze' state value approximation
        self.weights += total_weight_update
        
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
    '''
    def __init__(self, name, search_depth):
        super().__init__(name)
        self.search_depth = search_depth

    def action(self, board, trainer = None):
        """
        Method to select and place a worker, afterwards, place a building
        """
        board_levels, all_worker_coords = FastBoard.convert_board_to_array(board)
        fast_board = FastBoard()
        if trainer != None:
            if isinstance(trainer, RootStrapAB):
                minimax_tree = MinimaxWithPruning(board_levels, all_worker_coords, self.name, self.search_depth, fast_board, trainer.weights)
            elif isinstance(trainer, TreeStrapMinimax):
                minimax_tree = Minimax(board_levels, all_worker_coords, self.name, self.search_depth, fast_board, trainer.weights)

            new_board_levels, new_worker_coords = minimax_tree.get_best_node()
            new_board = FastBoard.convert_array_to_board(board, new_board_levels, new_worker_coords)
            #update weights if in training mode. instance must be called treestrap
            trainer.update_weights(minimax_tree, board_levels, all_worker_coords)
        else:
            minimax_tree = MinimaxWithPruning(board_levels, all_worker_coords, self.name, self.search_depth, fast_board)
            new_board_levels, new_worker_coords = minimax_tree.get_best_node()
            new_board = FastBoard.convert_array_to_board(board, new_board_levels, new_worker_coords)
        return new_board

def training_loop(trainer_a, trainer_b, agent_a, agent_b, n_iterations):
    a_wins = 0
    b_wins = 0
    for i in range(n_iterations):
        win = run_santorini(agent_a, agent_b, False, trainer_a, trainer_b)
        if win == 'A':
            a_wins += 1
        elif win == 'B':
            b_wins += 1
        print(trainer_a) #, trainer_b)
        print(f'{i+1}/{n_iterations} games completed. A has won {a_wins}/{i+1} games while B has won {b_wins}/{i+1} games.')

rootstrap = RootStrapAB()
treestrap = TreeStrapMinimax([-27.8177124,  -17.38723488,  -8.1350203,   -2.91982269, -12.06854257, -37.34683681, -24.54237358, -23.99931087])
    #[-36.87386823, -15.7237439,   52.04003867,  13.61611649,  20.9258601, -36.89773159,  -5.50453141,  10.62089711])
    #[-0.69692801,  0.85068979,  9.44413328,  0.45168497,  2.3968307,  -5.25062061, 0.02590619, 3.0396221])
agent_a = LinearRlAgentV2('A', 2)
agent_b = LinearRlAgentV2('B', 2)
training_loop(treestrap, None, agent_a, agent_b, 100)

#should I tie opposite features to each other..hmmm...or break them down further.....