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
    def __init__(self, board_levels, all_worker_coords, weights):
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

    def get_features(self):
        return np.array(self.calculate_position_features())

    def calculate_state_value(self):
        '''
        input: game board object, weights
        output: numerical value of given game state
        utilizes weights + board state to calculate the state value
        '''
        position_features = np.array(self.calculate_position_features())
        state_value = np.sum(position_features*self.weights)

        #ensures approximated value is within -9999 and 9999.
        state_value = min(state_value, 9990)
        state_value = max(-9990, state_value)
        return state_value

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

class LinearFnApproximatorV2():
    def __init__(self, board_levels, all_worker_coords, weights, fast_board):
        self.weights = weights
        self.BOARD_LEVELS = board_levels
        self.ALL_WORKER_COORDS = all_worker_coords
        self.A_WORKER_COORDS = all_worker_coords[:2]
        self.B_WORKER_COORDS = all_worker_coords[2:]
        self.NUM_WORKERS = 2
        self.CENTER_ROW, self.CENTER_COL = 2,2 #center coords
        self.BOARD_SIZE = 5
        self.MAX_POSSIBLE_MOVES = 100 #not proven, but close enough
        self.fast_board = fast_board
        self.features = None
        self.state_value = self.calculate_state_value()

    def __repr__(self):
        '''
        prints values of calculated features for debugging purposes
        '''
        position_features = self.calculate_position_features()
        mobility_features = self.calculate_mobility_features()
        return f'{self.features}'
        # return f'\These are the position features: {position_features}\
        # These are the mobility features {mobility_features}\
        # \n the value of this state is {self.state_value}'

    def get_features(self):
        #this is for the RL training
        return self.features

    def calculate_state_value(self):
        '''
        input: game board object, weights
        output: numerical value of given game state
        utilizes weights + board state to calculate the state value
        '''
        position_features = self.calculate_position_features()
        mobility_features = self.calculate_mobility_features()
        feature_vector = np.array(position_features + mobility_features)
        feature_vector = np.outer(feature_vector, feature_vector).flatten() #square it
        self.features = feature_vector
        state_value = np.sum(feature_vector*self.weights)

        #ensures approximated value is within -9999 and 9999.
        state_value = min(state_value, 9990)
        state_value = max(-9990, state_value)
        return state_value

    def calculate_position_features(self):
        '''
        input: self
        output: python list with value of each position-related feature
        Feature List:
        1. Worker Height of A1: Level 0
        2. Worker height of A1: Level 1
        3. Worker Height of A1: Level 2
        4. Worker A1 Distance from Centre
        5-8. Repeat for A2
        9-16. Repeat for B1, B2
        features are normalized from 0 to 1
        17. Distance A1, A2
        18. Distance A1, B1
        19. Distance A1, B2
        20. Distance A2, B1
        21. Distance A2, B2
        22. Distance B1, B2
        '''
        features = []

        def distance(x1, x2, y1, y2):
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        #calculate features 1 - 16:
        centre_dist_normalization_factor = distance(self.BOARD_SIZE-1, self.CENTER_ROW, self.BOARD_SIZE-1, self.CENTER_COL)
        for worker_row, worker_col in self.ALL_WORKER_COORDS:
            worker_height = self.BOARD_LEVELS[worker_row][worker_col]
            for level in [0,1,2]:
                if worker_height == level:
                    features.append(1)
                else:
                    features.append(0)
            worker_centre_dist = distance(worker_row, self.CENTER_ROW, worker_col, self.CENTER_COL)
            features.append(worker_centre_dist/centre_dist_normalization_factor)

        #calculate features 17-22:
        max_dist_normalization_factor = distance(self.BOARD_SIZE-1, 0, self.BOARD_SIZE-1, 0)
        for worker_1, worker_2 in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            worker_1_x, worker_1_y = self.ALL_WORKER_COORDS[worker_1]
            worker_2_x, worker_2_y = self.ALL_WORKER_COORDS[worker_2]
            relative_dist = distance(worker_1_x, worker_2_x, worker_1_y, worker_2_y)
            features.append(relative_dist/max_dist_normalization_factor)
        return features

    def calculate_mobility_features(self):
        '''
        input: self
        output: python list with value of each mobility-related feature
        Feature List:
        1. number of valid neighbouring squares for A1
        2. 1 if no squares for A1 to move to, 0 otherwise (binary)
        3. neighbouring level 0s for A1
        4. neighbouring level 1s for A1
        5. neighbouring level 2s for A1
        6. neighbouring level 3s for A1
        7. neighbouring level 4s for A1
        8-14. repeat for A2
        15-28. repeat for B1, B2
        # 29. overlapping level 0s for Player A
        # 30. overlapping level 1s for Player A
        # 31. overlapping level 2s for Player A
        # 32. overlapping level 3s for Player A
        # 33. overlapping level 4s for Player A
        # 34-38. repeat for player B
        features are normalized from 0 to 1
        '''
        features = []
        #features 1 to 28
        max_valid_neighbours = 8
        valid_neighbours_for_all_workers = []
        for worker_coords in self.ALL_WORKER_COORDS:
            valid_neighbours = self.fast_board.retrieve_valid_worker_moves(self.BOARD_LEVELS, self.ALL_WORKER_COORDS, worker_coords)
            #valid_neighbours_for_all_workers.append(valid_neighbours)

            num_valid_neighbours = len(valid_neighbours)
            #feature 1
            features.append(num_valid_neighbours/max_valid_neighbours) 
            #feature 2
            if num_valid_neighbours == 0: 
                features.append(1)
            else:
                features.append(0)
            
            #features 3-7
            neighbour_levels = [0 for i in range(5)]
            all_neighbours = self.fast_board.valid_coord_dict[worker_coords]
            for neighbour_row, neighbour_col in all_neighbours:
                neighbour_coord_level = self.BOARD_LEVELS[neighbour_row][neighbour_col]
                neighbour_levels[neighbour_coord_level] += 1
            neighbour_levels = [num/max_valid_neighbours for num in neighbour_levels] #normalize    
            features.extend(neighbour_levels)
        
        #features 29-38
        max_overlapping_neighbours = 4
        for player_workers in (self.A_WORKER_COORDS, self.B_WORKER_COORDS):
            overlap_counter = [0 for i in range(5)]
            worker_1_neighbours = self.fast_board.valid_coord_dict[player_workers[0]]
            worker_2_neighbours = self.fast_board.valid_coord_dict[player_workers[1]]
            overlapping_coords = set(worker_1_neighbours).intersection(worker_2_neighbours)
            for row,col in overlapping_coords:
                overlapping_coord_level = self.BOARD_LEVELS[row][col]
                overlap_counter[overlapping_coord_level] += 1
            overlap_counter = [num/max_overlapping_neighbours for num in overlap_counter] #normalize
            features.extend(overlap_counter)
        return features
            
class Minimax():
    '''
    Constructs entire Minimax Tree
    Inputs: Board object, current_player ('A' or 'B'), depth to search to, approximator_type('V1' or 'V2')
    '''
    def __init__(self, board_levels, all_worker_coords, current_player, depth, fast_board, weights, approximator_type):
        #initialize attributes
        self.depth = depth
        self.board_levels = board_levels
        self.all_worker_coords = all_worker_coords
        self.current_player = current_player
        self.weights = weights
        self.fast_board = fast_board
        self.approximator_type = approximator_type
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
            if self.approximator_type == 'V1':
                self.value = LinearFnApproximator(board_levels, all_worker_coords, self.weights).state_value
            elif self.approximator_type == 'V2':
                self.value = LinearFnApproximatorV2(board_levels, all_worker_coords, self.weights, self.fast_board).state_value
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
            
        return (f'This is a tree with depth {self.depth} and {len(self.child_nodes)} child nodes.\
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
            self.value = 9999
        elif self.winner == 'B':
            self.value = -9999

    def get_minimax_from_children(self):
        '''
        returns minimax values of child_nodes based on recursive minimax algorithm incorporating alpha-beta pruning
        '''
        if self.maximizing_player:
            maxValue = -math.inf
            for altered_board_levels, altered_worker_coords in self.possible_states:
                child_node = Minimax(altered_board_levels, altered_worker_coords, self.next_player, self.depth-1, self.fast_board, self.weights, self.approximator_type)
                self.child_nodes.append(child_node)
                value = child_node.value
                maxValue = max(maxValue, value)
            return maxValue
        else:
            minValue = math.inf
            for altered_board_levels, altered_worker_coords in self.possible_states:
                child_node = Minimax(altered_board_levels, altered_worker_coords, self.next_player, self.depth-1, self.fast_board, self.weights, self.approximator_type)
                self.child_nodes.append(child_node)
                value = child_node.value
                minValue = min(minValue, value)
            return minValue

    def get_best_node(self):
        for node in self.child_nodes:
            if self.value == node.value:
                return (node.board_levels, node.all_worker_coords)

class MinimaxWithPruning(Minimax):
    '''
    Core algorithm referenced from: https://www.youtube.com/watch?v=l-hh51ncgDI
    Constructs Minimax Tree with Alpha-Beta pruning
    Inputs: Board object, current_player ('A' or 'B'), depth to search to
    '''
    def __init__(self, board_levels, all_worker_coords, current_player, depth, fast_board, weights, approximator_type, alpha = -math.inf, beta = math.inf):
        self.alpha = alpha
        self.beta = beta
        super().__init__(board_levels, all_worker_coords, current_player, depth, fast_board, weights, approximator_type)

    def __repr__(self):
        total_2nd_order_nodes = 0
        for node in self.child_nodes:
            total_2nd_order_nodes += len(node.child_nodes)
            
        return (f'This is a pruned tree with depth {self.depth} and {len(self.child_nodes)} child nodes.\
        \n Current player is {self.current_player}\
        \n We have {total_2nd_order_nodes} 2nd order nodes')

    def get_minimax_from_children(self):
        '''
        returns minimax values of child_nodes based on recursive minimax algorithm incorporating alpha-beta pruning
        '''
        if self.maximizing_player:
            maxValue = -math.inf
            for altered_board_levels, altered_worker_coords in self.possible_states:
                child_node = MinimaxWithPruning(altered_board_levels, altered_worker_coords, self.next_player, self.depth-1, self.fast_board, self.weights, self.approximator_type, self.alpha, self.beta)
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
                child_node = MinimaxWithPruning(altered_board_levels, altered_worker_coords, self.next_player, self.depth-1, self.fast_board, self.weights, self.approximator_type, self.alpha, self.beta)
                self.child_nodes.append(child_node)
                value = child_node.value
                minValue = min(minValue, value)
                self.beta = min(self.beta, value)
                if self.beta <= self.alpha:
                    break
            return minValue
