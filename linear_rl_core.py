from Game import *
import numpy as np
import random, math

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
    def __init__(self, board, weights = [1,-1,1,2,4,-1,-2,-4,0,2,4,0,-2,-4,-1,1], fast_features_only = True):
        self.weights = weights
        self.BOARD = board
        self.PLAYER_CODES = ["A","B"]
        self.PLAYERS = [self.BOARD.PlayerA, self.BOARD.PlayerB]
        self.NUM_WORKERS = 2
        self.CENTER_X, self.CENTER_Y = 2,2 #center coords
        self.BOARD_SIZE = 5
        self.MAX_POSSIBLE_MOVES = 100 #not proven, but close enough
        self.fast_features_only = fast_features_only
        self.state_value = self.calculate_state_value()
        
    def __repr__(self):
        '''
        prints values of calculated features for debugging purposes
        '''
        mobility_features = self.calculate_mobility_features()
        position_features = self.calculate_position_features()
        return f'Here are the mobility features: {mobility_features}\
        \n and the position features: {position_features}\
        \n the value of this state is {self.state_value}'

    def calculate_state_value(self):
        '''
        input: game board object, weights
        output: numerical value of given game state
        utilizes weights + board state to calculate the state value
        '''
        position_features = np.array(self.calculate_position_features())
        if self.fast_features_only:
            return np.sum(position_features*self.weights[8:])
        else:
            mobility_features = self.calculate_mobility_features()
            feature_vector = np.array(mobility_features+position_features)
            return np.sum(feature_vector*self.weights)

    def calculate_mobility_features(self):
        '''
        input: game board object, player name, opponent name
        output: numpy array with value of each mobility related feature
        list of features:
        1. possible moves that your workers can collectively make
        2. possible moves that your opponent's workers can collectively make (approx. from curr state)
        3. number of moves that allow going from 0->1
        4. number of moves that allow going from 1->2
        5. number of moves that allow going from 2->3
        6,7,8. likewise for opponent.

        features normalized from 0 to 1
        '''
        playerA_possible_moves = self.BOARD.all_possible_next_states(self.PLAYERS[0].name)
        playerB_possible_moves = self.BOARD.all_possible_next_states(self.PLAYERS[1].name)
        possible_moves_normalization = self.MAX_POSSIBLE_MOVES
        features = []

        #feature 1 and 2
        for possible_moves in [playerA_possible_moves, playerB_possible_moves]:
            num_possible_moves = len(possible_moves)
            features.append(num_possible_moves/possible_moves_normalization)

        #features 3-8
        for player_code in self.PLAYER_CODES:
            for level in [0,1,2]:
                if player_code == 'A':
                    num_moves_upwards = self.num_possible_moves_upward(player_code, self.PLAYERS[0], playerA_possible_moves, level)
                    features.append(num_moves_upwards/possible_moves_normalization)
                else:
                    num_moves_upwards = self.num_possible_moves_upward(player_code, self.PLAYERS[1], playerB_possible_moves, level)
                    features.append(num_moves_upwards/possible_moves_normalization)
        return features

    def num_possible_moves_upward(self, player_code, player, player_possible_moves, start_level):
        '''
        inputs: current player, player obj, player's possible moves, amd desired starting level (either 0,1 or 2)
        outputs: number of possible moves up that level
        '''
        count = 0
        if self.num_workers_on_level(player, start_level) > 0:
            start_lvl_count_bef_move = self.num_workers_on_level(player, start_level)
            upper_lvl_count_bef_move = self.num_workers_on_level(player, start_level+1)
            for move in player_possible_moves:
                if player_code == 'A':
                    player_aft_move = move.PlayerA
                else:
                    player_aft_move = move.PlayerB
                start_lvl_count_aft_move = self.num_workers_on_level(player_aft_move, start_level)
                upper_lvl_count_aft_move = self.num_workers_on_level(player_aft_move, start_level+1)
                if (start_lvl_count_aft_move - start_lvl_count_bef_move == -1) and \
                    (upper_lvl_count_aft_move - upper_lvl_count_bef_move == 1):
                    count += 1
        return count

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
        for player in self.PLAYERS:
            for level in [0,1,2]:
                features.append(self.num_workers_on_level(player, level)/worker_normalization_factor)
        
        #calculate features 7, 8
        def distance(x1, x2, y1, y2):
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        for player in self.PLAYERS:
            total_dist = 0
            for worker in player.workers:
                worker_x, worker_y = worker.current_location[0], worker.current_location[1]
                total_dist += distance(worker_x, self.CENTER_X, worker_y, self.CENTER_Y)
            distance_normalization_factor = self.NUM_WORKERS*distance(self.BOARD_SIZE-1, self.CENTER_X, self.BOARD_SIZE-1, self.CENTER_Y)
            features.append(total_dist/distance_normalization_factor)
        
        return features

    def num_workers_on_level(self, player, level):
        '''
        inputs: player object, building level
        output: number of given player's workers on the given building level at that state
        '''
        output = 0
        for worker in player.workers:
            if worker.building_level == level:
                output += 1
        return output

class SearchTree():
    '''
    Parameters:
    board - board object, acts as root node for search tree
    current_player - either 'A' or 'B', depending on which is currently going to make a move
    depth 
    - how deep should the search tree go (each move = one lvl deeper regardless of player)
    - even number makes more sense, so that terminal nodes are at same player acting again)
    - if a player wins, search tree terminates there and self.winner is set to 'A' or 'B'
    
    each SearchTree contains a root_node, as well as all child node objects extending from it
    (basically more SearchTree objects)
    '''

    def __init__(self, board, current_player, depth): 
        self.depth = depth
        self.root_node = board
        self.current_player = current_player
        if current_player == 'A':
            self.current_player_name = board.PlayerA.name
        else:
            self.current_player_name = board.PlayerB.name

        if current_player == 'A':
            self.next_player = 'B'
        else:
            self.next_player = 'A'
        #check if winning node for previous player
        self.winner = self.check_previous_player_win()

        #generate list of child nodes
        if depth>0 and self.winner == None:
            self.populate_child_nodes()
        else:
            self.child_nodes = []

    def check_previous_player_win(self):
        '''
        this function checks if the prev player has already won the game (i.e. worker on lvl 3)
        output: 'A' or 'B' if either won, else None
        '''
        if self.current_player == 'A':
            player_to_check = self.root_node.PlayerB
        else:
            player_to_check = self.root_node.PlayerA

        if self.root_node.end_turn_check_win(player_to_check) != None:
            return self.next_player #returns alphabet of winning player
        else:
            return None

    def populate_child_nodes(self):
        '''
        this function uses the find all possible moves to get all possible child nodes
        recursively calls SearchTree(new_board, next player, and depth-1) and adds to a list
        '''
        possible_states = self.root_node.all_possible_next_states(self.current_player_name)
        self.child_nodes = []
        if len(possible_states) == 0: #if no possible moves, then other player already wins.
            self.winner = self.next_player
        else:
            for child_state in possible_states:
                self.child_nodes.append(SearchTree(child_state, self.next_player, self.depth-1))

    def __repr__(self):
        total_2nd_order_nodes = 0
        total_a_wins = 0
        total_b_wins = 0
        for node in self.child_nodes:
            total_2nd_order_nodes += len(node.child_nodes)
            if node.winner == 'A':
                total_a_wins += 1
            elif node.winner == 'B':
                total_b_wins += 1
            
        return (f'This is a search tree with depth {self.depth} and {len(self.child_nodes)} child nodes.\
        \n Current player is {self.current_player}\
        \n We have {total_2nd_order_nodes} 2nd order nodes\
        \n winning moves for A: {total_a_wins}, winning moves for B: {total_b_wins}')

class MinimaxTree(SearchTree):
    '''
    adds minimax values to indiv nodes of search tree, based on search depth.
    When not at win state (this needs to be checked with functions), use linear fn approximator
    Otherwise, if win state set value to either inf or -inf
    '''
    def __init__(self, board, current_player, depth):
        super().__init__(board, current_player, depth)
        self.value = 0
        if self.winner != None:
            if self.winner == 'A':
                self.value = math.inf
            elif self.winner == 'B':
                self.value = -math.inf
        elif depth !=0:
            #calculate minimax values from nodes
            node_values = []
            for node in self.child_nodes:
                node_values.append(node.value)
                if self.current_player == 'A':
                    self.value = max(node_values)
                else:
                    self.value = min(node_values)
        else: #winner == None and depth == 0
            self.value = LinearFnApproximator(self.root_node).state_value


    def minimax_from_child_nodes(self):
        pass

    def __repr__(self):
        return f'Minimax value is {self.value}'

    def populate_child_nodes(self):
        #find a way to reference searchtree function
        '''
        this function uses the find all possible moves to get all possible child nodes
        recursively calls SearchTree(new_board, next player, and depth-1) and adds to a list
        '''
        possible_states = self.root_node.all_possible_next_states(self.current_player_name)
        self.child_nodes = []
        if len(possible_states) == 0: #if no possible moves, then other player already wins.
            self.winner = self.next_player
        else:
            for child_state in possible_states:
                self.child_nodes.append(MinimaxTree(child_state, self.next_player, self.depth-1))

#finally, need another class for the RL algorithm itself (TD lambda? to train, 
#does the RL algorthim need some kind of regularization to prevent the rewards getting too ridiculous?

class MinimaxWithPruning():
    '''
    Core algorithm referenced from: https://www.youtube.com/watch?v=l-hh51ncgDI
    Constructs Minimax Tree with Alpha-Beta pruning
    Inputs: Board object, current_player ('A' or 'B'), depth to search to
    '''
    def __init__(self, board, current_player, depth, alpha = -math.inf, beta = math.inf):
        #initialize attributes
        self.depth = depth
        self.game_state = board
        self.current_player = current_player
        self.alpha = alpha
        self.beta = beta
        if current_player == 'A':
            self.current_player_name = board.PlayerA.name
            self.next_player = 'B'
            self.maximizing_player = True
        else:
            self.current_player_name = board.PlayerB.name
            self.next_player = 'A'
            self.maximizing_player = False

        #check if winning node for previous player
        self.winner = self.check_previous_player_win()        
        self.child_nodes = []
        #calculate value depending on situation
        if self.winner != None:
            self.set_win_node_value()
        elif depth == 0:
            self.value = LinearFnApproximator(self.game_state).state_value
        else:
            self.possible_states = self.game_state.all_possible_next_states(self.current_player_name)
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
        if self.current_player == 'A':
            player_to_check = self.game_state.PlayerB
        else:
            player_to_check = self.game_state.PlayerA

        if self.game_state.end_turn_check_win(player_to_check) != None:
            return self.next_player #returns alphabet of winning player
        else:
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
            for child_state in self.possible_states:
                child_node = MinimaxWithPruning(child_state, self.next_player, self.depth-1, self.alpha, self.beta)
                self.child_nodes.append(child_node)
                value = child_node.value
                maxValue = max(maxValue, value)
                self.alpha = max(self.alpha, value)
                if self.beta <= self.alpha:
                    break
            return maxValue
        else:
            minValue = math.inf
            for child_state in self.possible_states:
                child_node = MinimaxWithPruning(child_state, self.next_player, self.depth-1, self.alpha, self.beta)
                self.child_nodes.append(child_node)
                value = child_node.value
                minValue = min(minValue, value)
                self.beta = min(self.beta, value)
                if self.beta <= self.alpha:
                    break
            return minValue

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

class LinearRlAgent(RandomAgent):
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
        minimax = MinimaxWithPruning(board, self.name, 2)
        value = minimax.value
        for node in minimax.child_nodes:
            if minimax.value == node.value:
                return node.game_state

#Work in progress
    #the issue of whether to reduce the strength of the approximator in favour of greater search depth
    #linearRl agent not ideal way of taking actions
    #when generating possible moves, we want to prioritize moves we think will be good to speed up pruning
    #how to factor in rewards when minimax tree returns infinity...