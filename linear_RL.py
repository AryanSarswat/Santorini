from Game import *
import random, math
import numpy as np

#figure out out to organize classes in separate files later
class SearchTree():
    '''
    Parameters:
    board - board object, acts as root node for search tree
    current_player - either 'A' or 'B', depending on which is currently going to make a move
    depth 
    - how deep should the search tree go (each move = one lvl deeper regardless of player)
    - even number makes more sense, so that terminal nodes are at same player acting again)
    
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
        #generate list of child nodes
        if depth>0:
            self.populate_child_nodes()

    def populate_child_nodes(self):
        '''
        this function uses the find all possible moves to get all possible child nodes
        recursively calls SearchTree(new_board, next player, and depth-1) and adds to a list
        '''
        possible_states = self.root_node.all_possible_next_states(self.current_player_name)
        self.child_nodes = []
        for child_state in possible_states:
            self.child_nodes.append(SearchTree(child_state, self.next_player, self.depth-1))

    def __repr__(self):
        total_2nd_order_nodes = 0
        for node in self.child_nodes:
            total_2nd_order_nodes += len(node.child_nodes)

        return (f'This is a search tree with depth {self.depth} and {len(self.child_nodes)} child nodes.\
        \n Current player is {self.current_player}\
        \n We have {total_2nd_order_nodes} 2nd order nodes')

class MinimaxTree(SearchTree):
    '''
    adds minimax values to indiv nodes of search tree, based on search depth.
    When not at win state (this needs to be checked with functions), use linear fn approximator
    Otherwise, if win state set value to math.inf to ensure this gets picked
    '''

#finally, need another class for the RL algorithm itself (TD lambda? to train, 

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

class LinearFnApproximator():
    '''
    linear function approximator class that has two main methods:
    1. a means of approximating the state value from states using predetermined features
        - involves running allpossiblemoves from given board state; meant to be used on SearchTree terminal nodes
    2. a means of updating the given weights using a selected RL training method? (or do this separately...)
    
    inputs: board object, weights vector (1-dim numpy array), current_player ('A' or 'B')
    attributes: state_value (positive for A, negative for B)
    '''
    def __init__(self, board, weights = [1 for i in range(8)]):
        self.board = board
        self.weights = weights
        self.players = [self.board.PlayerA, self.board.PlayerB]
        self.NUM_WORKERS = 2
        self.CENTER_X, self.CENTER_Y = 2,2 #center coords
        self.BOARD_SIZE = 5
        self.MAX_POSSIBLE_MOVES = 100 #not proven, but close enough
        #self.state_value = self.calculate_state_value(self)

    def __repr__(self):
        position_features = self.calculate_position_features()
        mobility_features = self.calculate_mobility_features()
        return f'Here are the position features: {position_features}\
        \n and the mobility features: {mobility_features}'

    def calculate_state_value(self):
        '''
        input: game board object, weights
        output: numerical value of given game state
        utilizes weights + board state to calculate the state value
        '''
        mobility_features = self.calculate_mobility_features()
        position_features = self.calculate_position_features()
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
        playerA_possible_moves = self.board.all_possible_next_states(self.players[0].name)
        playerB_possible_moves = self.board.all_possible_next_states(self.players[1].name)
        features = []

        #feature 1 and 2
        for possible_moves in [playerA_possible_moves, playerB_possible_moves]:
            num_possible_moves = len(possible_moves)
            possible_moves_normalization = self.MAX_POSSIBLE_MOVES
            features.append(num_possible_moves/possible_moves_normalization)

        #features 3-8
        for player in range(2):
            for level in [0,1,2]:
                features.append(self.num_possible_moves_upward)

        return features

    def num_possible_moves_upward(self, board, player, player_possible_moves, level):
        '''
        inputs: current board state, player obj, player's possible moves, amd desired starting level (either 0,1 or 2)
        outputs: number of possible moves up that level
        '''
        if any workers on desired starting level: (using num_workers_on_level)
            for each move in player_possible_moves:
                compare using num_workers_on_level, then
                return the result
        else:
            return 0

        #find out max possible number of upward moves...is it also 100?

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
        for player in self.players:
            for level in [0,1,2]:
                features.append(self.num_workers_on_level(player, level))
        
        #calculate features 7, 8
        def distance(x1, x2, y1, y2):
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        for player in self.players:
            total_dist = 0
            for worker in player.workers:
                worker_x, worker_y = worker.current_location[0], worker.current_location[1]
                total_dist += distance(worker_x, self.CENTER_X, worker_y, self.CENTER_Y)
            distance_normalization_factor = self.NUM_WORKERS*distance(self.BOARD_SIZE-1, self.CENTER_X, self.BOARD_SIZE-1, self.CENTER_Y)
            features.append(total_dist/distance_normalization_factor)
        
        return features

    def num_workers_on_level(self, player, level):
        '''
        inputs: player, building level
        output: number of given player's workers on the given building level at that state
        '''
        output = 0
        for worker in player.workers:
            if worker.building_level == level:
                output += 1
        normalization_factor = self.NUM_WORKERS
        return output/normalization_factor

class LinearRlAgent(RandomAgent):
    '''
    basic RL agent using a linear function approximator and TD learning
    epsilon greedy policy too?
    '''
    def __init__(self, name):
        super().__init__(name)

###Code for Running Game
def run_santorini(agent1 = RandomAgent("A"), agent2 = RandomAgent("B")):
    '''
    should run a game of Santorini, allow choice of AI/human players
    '''
    board = Board(agent1, agent2)
    win = None
    #initial worker placement
    board = board.PlayerA.place_workers(board)
    board = board.PlayerB.place_workers(board)
    current_player = 'A'
    
    def get_current_board_player(current_player):
        if current_player == 'A':
            return board.PlayerA
        else:
            return board.PlayerB

    #game loop
    while win == None:

        board_player = get_current_board_player(current_player)
        win = board.start_turn_check_win(board_player)
        if win != None:
            break
        else:
            #print(SearchTree(board, current_player, depth=2))
            print(LinearFnApproximator(board))
            board.print_board()
            print("----------------------------------------------------------------\n")
            board = board_player.action(board)
            #because the board has been replaced, need to retrieve player obj again
            board_player = get_current_board_player(current_player)
            win = board.end_turn_check_win(board_player)
            if win != None:
                board.print_board()
                break
        
        if current_player == 'A':
            current_player = 'B'
        else:
            current_player = 'A'
        
    return win
    
run_santorini()