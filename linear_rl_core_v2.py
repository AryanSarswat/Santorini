from Game import *
import numpy as np
import random, math
from fast_board import FastBoard

###Rewritten to use fast_board game implementation for Minimax search

class LinearFnApproximator():
    '''
    linear function approximator class that has two main methods:
    1. a means of approximating the state value from states using predetermined features
        - involves running allpossiblemoves from given board state; meant to be used on SearchTree terminal nodes
    2. a means of updating the given weights using a selected RL training method? (or do this separately...)
    
    inputs: board object, weights vector (1-dim numpy array),
    whether to use only features that avoid invoking all_possible_moves)
    attributes: state_value (positive for A, negative for B)
    '''
    def __init__(self, board_levels, all_worker_coords, weights = [0,2,4,0,-2,-4,-1,1]):
        self.weights = weights
        self.BOARD_LEVELS = board_levels
        self.ALL_WORKER_COORDS = all_worker_coords
        self.A_WORKER_COORDS = all_worker_coords[:2]
        self.B_WORKER_COORDS = all_worker_coords[2:]
        self.NUM_WORKERS = 2
        self.CENTER_ROW, self.CENTER_COL = 2,2 #center coords
        self.BOARD_SIZE = 5
        self.MAX_POSSIBLE_MOVES = 100 #not proven, but close enough
        self.state_value = self.calculate_state_value()
        
    def __repr__(self):
        '''
        prints values of calculated features for debugging purposes
        '''
        position_features = self.calculate_position_features()
        return f'\These are the position features: {position_features}\
        \n the value of this state is {self.state_value}'

    def calculate_state_value(self):
        '''
        input: game board object, weights
        output: numerical value of given game state
        utilizes weights + board state to calculate the state value
        '''
        position_features = np.array(self.calculate_position_features())
        return np.sum(position_features*self.weights)

    def calculate_position_features(self):
        '''
        input: game board object
        output: python list with value of each position-related feature
        makes use of fact that playerA (+ve) in board is 1st player, playerB (-ve) is 2nd player
        list of features:
        1. # of workers on level 0
        2. # of workers on level 1
        3. # of workers on level 2
        4,5,6. repeat for Player2's workers
        7. piece distance from board centre (total?..if doing 2nd order interactions need to do by piece)
        8. repeat for Player2

        features are normalized from 0 to 1
        '''
        features = []

        #calculate features 1 - 6
        worker_normalization_factor = self.NUM_WORKERS
        for player_workers in [self.A_WORKER_COORDS, self.B_WORKER_COORDS]:
            for level in [0,1,2]:
                features.append(self.num_workers_on_level(player_workers, level)/worker_normalization_factor)
        
        #calculate features 7, 8
        def distance(x1, x2, y1, y2):
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        for player_workers in [self.A_WORKER_COORDS, self.B_WORKER_COORDS]:
            total_dist = 0
            for worker_row, worker_col in player_workers:
                total_dist += distance(worker_row, self.CENTER_ROW, worker_col, self.CENTER_COL)
            distance_normalization_factor = self.NUM_WORKERS*distance(self.BOARD_SIZE-1, self.CENTER_ROW, self.BOARD_SIZE-1, self.CENTER_COL)
            features.append(total_dist/distance_normalization_factor)
        return features

    def num_workers_on_level(self, player_workers, level):
        '''
        inputs: list containing coords of player's workers, building level
        output: number of given player's workers on the given building level at that state
        '''
        output = 0
        for worker_row, worker_col in player_workers:
            worker_building_level = self.BOARD_LEVELS[worker_row][worker_col]
            if worker_building_level == level:
                output += 1
        return output

#finally, need another class for the RL algorithm itself (TD lambda? to train, 
#does the RL algorthim need some kind of regularization to prevent the rewards getting too ridiculous?

class MinimaxWithPruning():
    '''
    Core algorithm referenced from: https://www.youtube.com/watch?v=l-hh51ncgDI
    Constructs Minimax Tree with Alpha-Beta pruning
    Inputs: Board object, current_player ('A' or 'B'), depth to search to
    '''
    def __init__(self, board_levels, all_worker_coords, current_player, depth, fast_board, alpha = -math.inf, beta = math.inf):
        #initialize attributes
        self.depth = depth
        self.board_levels = board_levels
        self.all_worker_coords = all_worker_coords
        self.current_player = current_player
        self.alpha = alpha
        self.beta = beta
        self.fast_board = fast_board
        if current_player == 'A':
            self.next_player = 'B'
            self.maximizing_player = True
            self.my_worker_coords = all_worker_coords[:2]
            self.opp_worker_coords = all_worker_coords[2:]
        elif current_player == 'B':
            self.next_player = 'A'
            self.maximizing_player = False
            self.my_worker_coords = all_worker_coords[2:]
            self.opp_worker_coords = all_worker_coords[:2]

        #check if winning node for previous player
        self.winner = self.check_previous_player_win()        
        self.child_nodes = []
        #calculate value depending on situation
        if self.winner != None:
            self.set_win_node_value()
        elif depth == 0:
            self.value = LinearFnApproximator(board_levels, all_worker_coords).state_value
        else:
            self.possible_states = fast_board.all_possible_next_states(board_levels, all_worker_coords, current_player)
            if len(self.possible_states) == 0: #if no possible moves, then other player already wins.
                self.winner = self.next_player
                self.set_win_node_value()
            else:
                self.value = self.get_minimax_from_children()

    def __repr__(self):
        total_2nd_order_nodes = 0
        for node in self.child_nodes:
            total_2nd_order_nodes += len(node.child_nodes)
            
        return (f'This is a pruned tree with depth {self.depth} and {len(self.child_nodes)} child nodes.\
        \n Current player is {self.current_player}\
        \n We have {total_2nd_order_nodes} 2nd order nodes')

    def check_previous_player_win(self):
        '''
        this function checks if the prev player has already won the game (i.e. worker on lvl 3)
        output: 'A' or 'B' if either won, else None
        '''
        for worker_row, worker_col in self.opp_worker_coords:
            if self.board_levels[worker_row][worker_col] == 3:
                return self.next_player #returns alphabet of winning player
        return None

    def set_win_node_value(self):
        '''
        depending on winning player, sets self.value either to positive or negative infinity
        '''
        if self.winner == 'A':
            self.value = math.inf
        elif self.winner == 'B':
            self.value = -math.inf

    def get_minimax_from_children(self):
        '''
        returns minimax values of child_nodes based on recursive minimax algorithm incorporating alpha-beta pruning
        '''
        if self.maximizing_player:
            maxValue = -math.inf
            for altered_board_levels, altered_worker_coords in self.possible_states:
                child_node = MinimaxWithPruning(altered_board_levels, altered_worker_coords, self.next_player, self.depth-1, self.fast_board, self.alpha, self.beta)
                self.child_nodes.append(child_node)
                value = child_node.value
                maxValue = max(maxValue, value)
                self.alpha = max(self.alpha, value)
                if self.beta <= self.alpha:
                    break
            return maxValue
        else:
            minValue = math.inf
            for altered_board_levels, altered_worker_coords in self.possible_states:
                child_node = MinimaxWithPruning(altered_board_levels, altered_worker_coords, self.next_player, self.depth-1, self.fast_board, self.alpha, self.beta)
                self.child_nodes.append(child_node)
                value = child_node.value
                minValue = min(minValue, value)
                self.beta = min(self.beta, value)
                if self.beta <= self.alpha:
                    break
            return minValue

    def get_best_node(self):
        for node in self.child_nodes:
            if self.value == node.value:
                return (node.board_levels, node.all_worker_coords)

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
    def __init__(self, name):
        super().__init__(name)

    def action(self, board):
        """
        Method to select and place a worker, afterwards, place a building
        """
        board_levels, all_worker_coords = FastBoard.convert_board_to_array(board)
        fast_board = FastBoard()
        minimax = MinimaxWithPruning(board_levels, all_worker_coords, self.name, 3, fast_board)
        new_board_levels, new_worker_coords = minimax.get_best_node()
        new_board = FastBoard.convert_array_to_board(board, new_board_levels, new_worker_coords)
        return new_board

#Work in progress
    #the issue of whether to reduce the strength of the approximator in favour of greater search depth
    #linearRl agent not ideal way of taking actions
    #when generating possible moves, we want to prioritize moves we think will be good to speed up pruning
    #how to factor in rewards when minimax tree returns infinity...