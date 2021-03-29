from Game import *
import numpy as np
import random
from linear_rl_core_v2 import RandomAgent, MinimaxWithPruning, LinearFnApproximator, LinearRlAgentV2
from fast_board import FastBoard
from linear_rl_launcher import run_santorini

class RootStrapABTrainer():
    '''
    performs minimax search at current state, then updates parameters using SGD
    to approximated value of node closer to minimax search value
    '''

    def __init__(self, weights = None, learning_rate = 10**-5):
        self.NUM_WEIGHTS = 8
        self.learning_rate = learning_rate
        if weights == None:
            #randomly initialize weights btwn -1 and 1
            self.weights = [random.uniform(-1,1) for i in range(self.NUM_WEIGHTS)]
        else:
            self.weights = weights

    def __repr__(self):
        return f'Current weights are {self.weights}'

    def update_weights(self, minimax_tree, board_levels, all_worker_coords):
        #update weights of approximator towards minimax search value
        linear_approximator = LinearFnApproximator(board_levels, all_worker_coords, self.weights)
        approximated_value = linear_approximator.state_value
        feature_vector = linear_approximator.get_features()
        error = minimax_tree.value - approximated_value
        weight_update = self.learning_rate * error * feature_vector
        self.weights += weight_update

class TreeStrapMinimax():
    pass

def training_loop():
    rootstrap = RootStrapABTrainer([-0.69692801,  0.85068979,  9.44413328,  0.45168497,  2.3968307,  -5.25062061, 0.02590619, 3.0396221])
    #[-0.43187563, -3.59073704, -0.70871379, -0.36054797, -0.41201018, -4.29358644, -3.73424819, -0.91657727])
    agent_a = LinearRlAgentV2('A', 2)
    agent_b = LinearRlAgentV2('B', 1)

    a_wins = 0
    b_wins = 0
    for i in range(100):
        win = run_santorini(agent_a, agent_b, False, trainer_a = None, trainer_b = None)
        if win == 'A':
            a_wins += 1
        elif win == 'B':
            b_wins += 1
        print(rootstrap)
        print(f'{i+1}/100 games completed. A has won {a_wins}/{i+1} games while B has won {b_wins}/{i+1} games.')

training_loop()

#should I tie opposite features to each other..hmmm...or break them down further.....