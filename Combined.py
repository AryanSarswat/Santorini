import numpy as np
import sys
from copy import deepcopy
import os
from random import shuffle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random, math
from fast_board import FastBoard

#dictionary containing board state codes. first number is player, second number is worker number
mappings = {
    (0,None) : 0,
    (1,None) : 1,
    (2,None) : 2,
    (3,None) : 3,
    (4,None) : 4,
    (0,'A') : 5,
    (1,'A') : 6,
    (2,'A') : 7,
    (3,'A') : 8,
    (0,'B') : 9,
    (1,'B') : 10,
    (2,'B') : 11,
    (3,'B') : 12,
}


def get_neighbours(coord):
    '''
    Returns a list of all neighbour location from a given state. Assumes 5x5 board.
    '''
    neighbour_locations = [[coord[0]+1,coord[1]],[coord[0],coord[1]+1],[coord[0]+1,coord[1]+1],[coord[0]-1,coord[1]],[coord[0],coord[1]-1],[coord[0]-1,coord[1]-1],[coord[0]+1,coord[1]-1],[coord[0]-1,coord[1]+1]]
    #Only keep coordinates within the board
    neighbour_locations = list(filter(lambda element: (element[0] >= 0 and element[0] < 5) and (element[1] >= 0 and element[1] < 5),neighbour_locations))
    return neighbour_locations

class Worker():
    '''
    Worker Class containing the 
    1) Name of Worker
    2) Current Location
    3) Previous Location
    4) Building Level of Worker
    '''
    def __init__(self,Worker_Loc,Name):
        self.name = Name
        self.current_location = Worker_Loc
        self.previous_location = Worker_Loc
        self.building_level = 0

    def update_location(self, newLocation):
        self.previous_location = self.current_location
        self.current_location = newLocation    
    
    def update_building_level(self, newLevel):
        self.building_level = newLevel

class Square():
    """
    Element used to represent a square on the board
    1) Contain the index of the row and column of the board
    2) Contains the building level
    3) Contains the worker if a worker is present else it contains None
    """
    def __init__(self,row,col):
        self.row = row
        self.col = col
        self.building_level = 0
        self.worker = None

    def add_building(self):
        '''
        Adds a building to the element
        '''
        #Check if Dome is there or not
        if self.building_level < 4 :
            self.building_level += 1
        else:
            raise Exception("Max Building Error Reached")
    
    def update_worker(self, worker):
        #Update Square if worker moves on the square
        if self.worker == None:
            self.worker = worker
        else:
            raise Exception("Worker Already Present in this Square")
        
    def remove_worker(self):
        # Removes worker from square
        self.worker = None

class HumanPlayer():
    '''
    Player class contains
    1) 2 distinguishable Worker pieces
    '''
    def __init__(self, name):
        self.name = name        
        self.workers = [Worker([], str(name)+"1"), Worker([], str(name)+"2")]

    
    def place_workers(self, board):
        """
        Method to place a player's worker on the board
        """
        place_count = 0
        while place_count < 2:
            try:
                #Input worker to move coordinates and format into list
                worker_to_place_coord = input(f"Player {self.name}, please enter the coordinates of the worker to place: ")
                worker_coord = list(map(lambda x: int(x),worker_to_place_coord.split(",")))
                # Updates worker and square
                self.workers[place_count].update_location(worker_coord)
                board.board[worker_coord[0]][worker_coord[1]].update_worker(self.workers[place_count])
                place_count += 1
            except Exception:
                print("Input error! Please enter coordinates of an empty square.")
                continue
        return board
    
    def action(self, board):
        """
        Method to select and place a worker, afterwards, place a building
        """
        # Selects worker to be moved
        worker_selection = input(f"Player {self.name}, please select the worker to be moved (1 or 2): ")
        while not ((worker_selection == "1" or worker_selection == "2") and len(worker_selection) == 1):
            print("Please enter a valid worker number")
            worker_selection = input(f"Player {self.name}, please select the worker to be moved (1 or 2): ")
        selected_worker = self.workers[int(worker_selection)-1]
        # Input the coordinates to move the selected worker
        new_coord = input(f"Player {self.name}, please enter the coordinates where worker {selected_worker.name} is to be placed: ")
        new_coord = list(map(lambda x: int(x),new_coord.split(",")))
        # Error handling for new_coordinates for the selected worker
        while new_coord not in board.possible_worker_movements(selected_worker.current_location):
            print("That is not a valid move location")
            new_coord = input(f"Player {self.name}, please the coordinates where worker {selected_worker.name} is to be placed: ")
            new_coord = list(map(lambda x: int(x),new_coord.split(",")))
        print("\n")
        # Updates worker and square
        old_coord_row, old_coord_column = selected_worker.current_location[0], selected_worker.current_location[1]
        board.board[old_coord_row][old_coord_column].remove_worker()
        selected_worker.update_location(new_coord)
        board.board[new_coord[0]][new_coord[1]].update_worker(selected_worker)
        selected_worker.update_building_level(board.board[new_coord[0]][new_coord[1]].building_level)
        board.print_board()
        print("----------------------------------------------------------------\n")
        # Check if the previous move was a winning move (bypasses building stage)
        if board.end_turn_check_win(self) != None:
            return board
        # Input the coordinates to build a building
        build_coord = input(f"Player {self.name}, Please enter coordinates of where you would like to build: ")
        build_coord = list(map(lambda x: int(x),build_coord.split(",")))
        # Error handling for the building coordinates
        while build_coord not in board.valid_building_options(selected_worker.current_location):
            print("That is not valid build location \n")
            build_coord = input(f"Player {self.name}, Please enter coordinates of where you would like to build: ")
            build_coord = list(map(lambda x: int(x),build_coord.split(",")))  
        print("----------------------------------------------------------------\n")
        board = board.update_building_level(build_coord)
        return board


class Board(object):
    '''
    Board class contains
    1) The entire board which is a list of Squares
    2) List of all workers for both players
    3) A list of all of Player 1's workers
    4) A list of all of Player 2's workers
    '''
    
    def __init__(self, agent1, agent2):
        """
        Constructor for training with 2 agents
        """
        self.board = []
        for i in range(5):
            temp_board = []
            for j in range(5):
                temp_board.append(Square(i,j))
            self.board.append(temp_board) 
    
        self.workers = []
        self.PlayerA = agent1
        self.PlayerB = agent2
        self.total_building_count = 0
    
    def print_board(self):
        '''
        Prints the board into a readable format
        '''
        for row in self.board:
            temp_list = []
            for square in row:
                if square.worker == None:
                    temp_list.append((square.building_level,square.worker))
                else:
                    temp_list.append((square.building_level,square.worker.name))
            print(temp_list)
            print("\n")

    def valid_building_options(self,location):
        '''
        Returns all the possible building options at certain location
        '''
        neighbour_locations = get_neighbours(location)
        possible_options = []
        for row,col in neighbour_locations:
                if self.board[row][col].building_level < 4  and (self.board[row][col].worker == None):
                    possible_options.append([row,col])
        return possible_options
    
    def update_worker_location(self,previous_location,new_location):
        '''
        Updates the board on where the worker is and also updates the Workers list on the location of the worker
        Returns New State containing the Update
        '''
        #Get Worker
        state = deepcopy(self)
        worker = state.board[previous_location[0]][previous_location[1]].worker
        #Update Worker
        worker.current_location = new_location
        worker.previous_location = previous_location
        worker.building_level = state.board[new_location[0]][new_location[1]].building_level
        #Update Location
        state.board[new_location[0]][new_location[1]].update_worker(worker)
        state.board[worker.previous_location[0]][worker.previous_location[1]].worker = None
        return state
    
    #Build a building
    def update_building_level(self,coord):
        '''
        Functionality to add a building level
        Returns a new state containing the updating building level
        '''
        state = deepcopy(self)
        state.board[coord[0]][coord[1]].add_building()
        state.total_building_count+=1
        return state

    def possible_worker_movements(self,Worker_loc):
        '''
        Returns all list of the possible movements for a worker
        '''
        possible_movements = []
        neighbour_locations = get_neighbours(Worker_loc)
        #Get Building level of the Worker's current postion before moving
        current_building_level = self.board[Worker_loc[0]][Worker_loc[1]].building_level
        possible_building_levels = []
        
        #Checks possible building levels
        if current_building_level == 0 :
            possible_building_levels.append(0)
            possible_building_levels.append(1)
        elif current_building_level == 1:
            possible_building_levels.append(current_building_level)
            possible_building_levels.append(current_building_level+1)
            possible_building_levels.append(current_building_level-1)
        elif current_building_level == 2:
            possible_building_levels.append(current_building_level)
            possible_building_levels.append(current_building_level+1)
            possible_building_levels.append(current_building_level-1)
            possible_building_levels.append(current_building_level-2)

        #Sort based on Building level
        possible_movements = list(filter(lambda element : self.board[element[0]][element[1]].building_level in possible_building_levels, neighbour_locations))
        #Sort based on Occupancy
        possible_movements  = list(filter(lambda element: self.board[element[0]][element[1]].worker == None , possible_movements))
        return possible_movements
    
    def all_possible_next_states(self, name):
        """
        Return a copy of all the possible next states after a worker movement and a bulding placement
        """
        if name == "A":
        #Movements for the workers of Player A
            worker_1_loc = self.PlayerA.workers[0].current_location
            worker_2_loc = self.PlayerA.workers[1].current_location
            possible_worker_1_movements = self.possible_worker_movements(worker_1_loc)
            possible_worker_2_movements = self.possible_worker_movements(worker_2_loc)
            possible_next_state_1 = []
            possible_next_state_2 = []
            #Make the move and return a deepcopy of the new state
            for action in possible_worker_1_movements:
                possible_next_state_1.append(self.update_worker_location(worker_1_loc,action))
            for action in possible_worker_2_movements:
                possible_next_state_2.append(self.update_worker_location(worker_2_loc,action))
            #Return the possible build options
            possible_builds_1 = [state.valid_building_options(state.PlayerA.workers[0].current_location) for state in possible_next_state_1]
            possible_builds_2 = [state.valid_building_options(state.PlayerA.workers[1].current_location) for state in possible_next_state_2]
            next_states = []
            #Apply build and return a deepcopy of the new state
            for n in range(len(possible_builds_1)):
                builds = possible_builds_1[n]
                for b in builds:
                    next_states.append(possible_next_state_1[n].update_building_level(b))
            for n in range(len(possible_builds_2)):
                builds = possible_builds_2[n]
                for b in builds:
                    next_states.append(possible_next_state_2[n].update_building_level(b))
        else:
            #Movements for the workers of Player B
            worker_1_loc = self.PlayerB.workers[0].current_location
            worker_2_loc = self.PlayerB.workers[1].current_location
            possible_worker_1_movements = self.possible_worker_movements(worker_1_loc)
            possible_worker_2_movements = self.possible_worker_movements(worker_2_loc)
            possible_next_state_1 = []
            possible_next_state_2 = []
            #Make the move and return a deepcopy of the new state
            for action in possible_worker_1_movements:
                possible_next_state_1.append(self.update_worker_location(worker_1_loc,action))
            for action in possible_worker_2_movements:
                possible_next_state_2.append(self.update_worker_location(worker_2_loc,action))
            #List of the possible build options
            possible_builds_1 = [state.valid_building_options(state.PlayerB.workers[0].current_location) for state in possible_next_state_1]
            possible_builds_2 = [state.valid_building_options(state.PlayerB.workers[1].current_location) for state in possible_next_state_2]
            next_states = []
            #Apply build and return a deepcopy of the new state
            for n in range(len(possible_builds_1)):
                builds = possible_builds_1[n]
                for b in builds:
                    next_states.append(possible_next_state_1[n].update_building_level(b))
            for n in range(len(possible_builds_2)):
                builds = possible_builds_2[n]
                for b in builds:
                    next_states.append(possible_next_state_2[n].update_building_level(b))
        return next_states

    def end_turn_check_win(self,Player):
        '''
        Checks if a player's workers have reached level 3
        '''
        if Player.name == "A":
            for worker in self.PlayerA.workers:
                if worker.building_level == 3:
                    return "A"
                else:
                    continue
        else:
            for worker in self.PlayerB.workers:
                if worker.building_level == 3:
                    return "B"
                else:
                    continue
        return None
    
    def start_turn_check_win(self,Player):
        '''
        Check at the start of the turn to see if other player has won due to no possible worker movements
        '''
        if Player.name == "A":
            pos_moves = []
            for worker in self.PlayerA.workers:
                location = worker.current_location
                pos_moves+=self.possible_worker_movements(location)
            if len(pos_moves) == 0:
                return "B"
            else:
                return None
        else:
            pos_moves = []
            for worker in self.PlayerB.workers:
                location = worker.current_location
                pos_moves+=self.possible_worker_movements(location)
            if len(pos_moves) == 0:
                return "A"
            else:
                return None
    
    def is_terminal(self):
        """
        Returns True if the game is over
        Return False if game is not over
        """
        Player = self.Player_turn()
        if Player == "A":
            if (self.start_turn_check_win(self.PlayerA) != None) or (self.end_turn_check_win(self.PlayerB) != None):
                return True
        elif Player == "B":
            if (self.start_turn_check_win(self.PlayerB) != None) or (self.end_turn_check_win(self.PlayerA) != None):
                return True        
        return False
 
    def reward(self):
        """
        Return 1 if PlayerA won
        Returns -1 if PlayerB won
        Return 0 if game is not over
        """
        if self.is_terminal():
            if (self.start_turn_check_win(self.PlayerB) != None) or (self.end_turn_check_win(self.PlayerA) != None):
                return 1
            else:
                return -1
        else:
            return 0
    
    def Player_turn(self):
        """
        Return A if it is Player A's turn
        Return B if it is Player B's Turn
        """
        if (self.total_building_count%2) == 0:
            return "A"
        else:
            return "B"



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
    def __init__(self, name,depth=3):
        super().__init__(name)
        self.depth = depth

    def action(self, board):
        """
        Method to select and place a worker, afterwards, place a building
        """
        board_levels, all_worker_coords = FastBoard.convert_board_to_array(board)
        fast_board = FastBoard()
        minimax = MinimaxWithPruning(board_levels, all_worker_coords, self.name, self.depth, fast_board)
        new_board_levels, new_worker_coords = minimax.get_best_node()
        new_board = FastBoard.convert_array_to_board(board, new_board_levels, new_worker_coords)
        return new_board

def state_mappings():
    mappings = {
        (0, None): 0,
        (1, None): 1,
        (2, None): 2,
        (3, None): 3,
        (4, None): 4,
        (0, 'A'): 5,
        (1, 'A'): 6,
        (2, 'A'): 7,
        (3, 'A'): 8,
        (0, 'B'): 9,
        (1, 'B'): 10,
        (2, 'B'): 11,
        (3, 'B'): 12,
    }
    return mappings


class Logger():
    def __init__(self):
        self.values = []


class ValueFunc(nn.Module):
    def __init__(self):
        super(ValueFunc, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, (3, 3), stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), stride=1)
        self.batch3 = nn.BatchNorm2d(64)
        self.flat = nn.Flatten()

        x = torch.randn(1, 2, 5, 5)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        self.optimizer = optim.SGD(self.parameters(), lr=0.1)
        self.loss = nn.MSELoss()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.epsilon = 0.999
        self.epsilon_min = 0.01

    def convs(self, x):
        x = x.float()
        x = self.conv1(x)
        x = F.relu(self.batch1(x))
        x = self.conv2(x)
        x = F.relu(self.batch2(x))
        x = self.conv3(x)
        x = F.relu(self.batch3(x))
        x = self.flat(x)

        if self._to_linear == None:
            self._to_linear = x.shape[1]
        return x

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            x = x.reshape(1, 2, 5, 5).to(self.device)
            x = self.convs(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        return x

    def convertTo2D(self, board):
        """
        Takes in a board and converts it into 2D tensor form with shape (2, 5, 5)
        """
        data = []
        buildings = []
        players = []
        for squares in board.board:
            temp_lst = []
            temp_lst2 = []
            for square in squares:
                if square.worker == None:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(0)
                elif square.worker.name[0] == self.name:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(1)
                else:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(-1)
            buildings.append(temp_lst)
            players.append(temp_lst2)
        data.append(buildings)
        data.append(players)
        return torch.as_tensor(data)

class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(in_features=325, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.value_head = nn.Linear(in_features=64, out_features=1)
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(),lr=1e-2)
        self.loss = nn.MSELoss()

    def forward(self, x):
        """
        Feed forward into the Neural Network
        """
        x = torch.from_numpy(x).float().to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value_logit = self.value_head(x)

        return torch.tanh(value_logit)
 
    def predict(self, board):
        """
        Predict the value of a state
        """
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            v = self.forward(board)

        return v.data.cpu().numpy()[0]


class Trainer():
    def __init__(self, args, NN=None):
        self.args = args
        self.state = Board(RandomAgent("A"), RandomAgent("B"))
        self.training_examples = []
        self.mcts = None
        self.nn = NN if NN != None else Neural_Network()
        self.loss_array = []
        self.mappings = {
            (0, None): 0,
            (1, None): 1,
            (2, None): 2,
            (3, None): 3,
            (4, None): 4,
            (0, 'A'): 5,
            (1, 'A'): 6,
            (2, 'A'): 7,
            (3, 'A'): 8,
            (0, 'B'): 9,
            (1, 'B'): 10,
            (2, 'B'): 11,
            (3, 'B'): 12,
        }
        self.nn.to(self.nn.device)

    def initialize_mcts(self):
        self.state.PlayerA.place_workers(self.state)
        self.state.PlayerB.place_workers(self.state)
        root = Node(self.state)
        self.mcts = MCTS(root, self.nn, self.args)

    def convert_nodes_to_training_data(self, set_of_nodes):
        training_data = [(i.state, i.value()) for i in set_of_nodes]
        shuffle(training_data)
        return training_data

    def generate_training_data(self):
        """
        Perform iteration of MCTS and return a collapsed tree for training
        """
        print("\nGenerating Data")
        training_data = []
        temp_MCTS = self.mcts
        node = self.mcts.root
        for i in tqdm(range(self.args['depth'])):
            root = temp_MCTS.breadth_run(node)
            app = list(temp_MCTS.collapse(root))
            training_data += app
            node = root.select_child()

        return training_data

    def learn(self, train_examples):
        """
        Learn using One MCTS tree
        """
        print("\nLearning from Data")
        boards = self.convert_nodes_to_input(train_examples)
        target_values = [node.value() for node in train_examples]
        data = [(boards[i], target_values[i]) for i in range(len(boards))]
        np.random.shuffle(data)

        for i in range(len(boards)):
            target = torch.tensor(
                data[i][1], dtype=torch.float32).to(self.nn.device)
            target = target.view(1)
            temp = torch.from_numpy(data[i][0]).float().to(self.nn.device)
            pred = self.nn.forward(temp).to(self.nn.device)
            loss = self.nn.loss(pred, target)
            self.nn.optimizer.zero_grad()
            loss.backward()
            self.nn.optimizer.step()
            self.loss_array.append(loss.item())

        self.plot_loss()

    def train(self):
        self.loss_array = []
        for i in tqdm(range(self.args["epochs"])):
            training_examples = self.generate_training_data()
            self.learn(training_examples)
        self.save_checkpoint(
            r'C:\Users\sarya\Documents\GitHub\Master-Procrastinator')
        pass

    def save_checkpoint(self, folder):
        """
        Save the Neural Network
        """
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, "MCTS_AI")
        torch.save(self.nn.state_dict(), filepath)

    def plot_loss(self):
        plt.plot(self.loss_array)
        plt.title("Loss versus iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()


class Trainer_CNN(Trainer):

    def __init__(self, player,args, NN=None):
        self.args = args
        self.state = Board(LinearRlAgentV2("A"), LinearRlAgentV2("B"))
        self.training_examples = []
        self.mcts = None
        self.nn = NN if NN != None else ValueFunc()
        self.loss_array = []
        self.mappings = {
            (0, None): 0,
            (1, None): 1,
            (2, None): 2,
            (3, None): 3,
            (4, None): 4,
            (0, 'A'): 5,
            (1, 'A'): 6,
            (2, 'A'): 7,
            (3, 'A'): 8,
            (0, 'B'): 9,
            (1, 'B'): 10,
            (2, 'B'): 11,
            (3, 'B'): 12,
        }
        self.nn.to(self.nn.device)
        self.name = player
        self.workers = [Worker([], str("A")+"1"), Worker([], str("A")+"2")]
        self.fast_board = FastBoard()

    def convertTo2D(self, board):
        """
        Takes in a board and converts it into 2D tensor form with shape (2, 5, 5)
        """
        data = []
        buildings = []
        players = []
        for squares in board.board:
            temp_lst = []
            temp_lst2 = []
            for square in squares:
                if square.worker == None:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(0)
                elif square.worker.name[0] == "A":
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(1)
                else:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(-1)
            buildings.append(temp_lst)
            players.append(temp_lst2)
        data.append(buildings)
        data.append(players)
        return torch.as_tensor(data)

    def convert_nodes_to_training_data(self, set_of_nodes):
        training_data = [(i.state, i.value()) for i in set_of_nodes]
        shuffle(training_data)
        return training_data

    def generate_training_data(self):
        """
        Perform iteration of MCTS and return a collapsed tree for training
        """
        print("\nGenerating Data")

        temp_MCTS = self.mcts
        node = self.mcts.root
        training_data = []
        """
        for i in tqdm(range(self.args['Iterations'])):
            temp_MCTS.run(node.state.Player_turn())
        training_data = temp_MCTS.collapse()
        """
        for i in tqdm(range(self.args["Num_Simulations"])):
            root = temp_MCTS.breadth_run(node)
            app = list(temp_MCTS.collapse(root))
            training_data+=app
            node = root.select_child()
        
        return training_data

    def save_checkpoint(self, folder):
        """
        Save the Neural Network
        """
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, "MCTS_AI_CNN")
        torch.save(self.nn.state_dict(), filepath)

    def learn(self, train_examples):
        """
        Learn using One MCTS tree
        """
        print("\nLearning from Data")

        for i in range(len(train_examples)):
            target = torch.tensor(
                train_examples[i][1], dtype=torch.float32).to(self.nn.device)
            target = target.view(1)
            converted_state = self.convertTo2D(train_examples[i][0])
            pred = torch.nn.forward(converted_state).to(t.nn.device)
            loss = self.nn.loss(pred, target)
            self.nn.optimizer.zero_grad()
            loss.backward()
            self.nn.optimizer.step()
            self.loss_array.append(loss.item())

        self.plot_loss()

    def train(self):
        self.loss_array = []
        for i in tqdm(range(self.args["epochs"])):
            training_examples = self.generate_training_data()
            training_examples = self.convert_nodes_to_training_data(training_examples)
            self.learn(training_examples)
        self.save_checkpoint(r'C:\Users\sarya\Desktop\Semester 4\ISM\Game')
        pass

    def action(self, board):
        build,worker = self.fast_board.convert_board_to_array(board)
        pos_states = self.fast_board.all_possible_next_states(build,worker,board.Player_turn())
        b_pos_states = [self.fast_board.convert_array_to_board(board,i,j) for i,j in pos_states]
        values = []
        for state in b_pos_states:
            converted_state = self.convertTo2D(state)
            values.append(torch.flatten(self.nn.forward(converted_state).to(self.nn.device)))
        if board.Player_turn() == "A":
            return b_pos_states[torch.argmax(torch.cat(values)).item()]
        else:
            return b_pos_states[torch.argmin(torch.cat(values)).item()]

    def place_workers(self, board):
        """
        Method to randomly place agent's workers on the board
        """
        place_count = 0
        while place_count < 2:
            try:
                coords = [np.random.randint(0, 5), np.random.randint(0, 5)]
                # Updates worker and square
                self.workers[place_count].update_location(coords)
                board.board[coords[0]][coords[1]].update_worker(
                    self.workers[place_count])
                place_count += 1
            except Exception:
                continue
        return board

def upper_confidence_bound(node):
    """
    Function which return the Upper Confidence Bound
    C = 2 has been implemented to balance exploratio and exploitation
    """
    if node.visit_count == 0:
        return np.inf
    value = node.value()
    explore = 2*np.sqrt(np.log(node.parent.visit_count)/node.visit_count)
    return value+explore

class Node():
    """
    Class representing the a node in the MCTS
    """
    def __init__(self,state,parent=None):
        self.state = state
        self.children = {}
        self.parent = parent
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = state.Player_turn()
        self.fast_board = FastBoard()
    
    def add_children(self,children):
        for child in children:
            self.children.add(child)
    
    def value(self):
        """
        Calculates the value
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum/self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

    def select_action(self,temperature):
        """
        Select an action based on visit count and temperature
        """
        visit_counts = np.array([child.visit_count for child in self.children.keys()])
        actions = list(self.children.keys())
        if temperature == 0:
            new_state = actions[np.argmax(visit_counts)]
        elif temperature == np.inf:
            new_state = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            new_state = np.random.choice(actions, p=visit_count_distribution)
        
        return new_state
    
    def select_child(self):
        """
        Selects the child with the highest UCB score
        """
        children_nodes = list(self.children.keys())
        UCB_score = list(map(upper_confidence_bound,children_nodes))
        return children_nodes[np.argmax(UCB_score)]
    
    def expand(self):
        """ 
        Expand Node
        """
        build,worker = self.fast_board.convert_board_to_array(self.state)
        children = self.fast_board.all_possible_next_states(build,worker,self.state.Player_turn())
        b_children = [self.fast_board.convert_array_to_board(self.state,i,j) for i,j in children]
        for child in b_children:
            self.children[Node(child,parent=self)] = child
        pass

class MCTS():
    """
    Class for containing the Monte Carlo Tree
    """
    def __init__(self,root,model,args):
        self.root = root
        self.model = model
        self.args = args
        self.depth = args["Tree_depth"]
        self.fast_board = FastBoard()

    def backpropagate(self,search_path,value,to_play):
        """
        Backpropagate the value of state
        """
        for node in reversed(search_path):
            val = 1 if node.to_play == to_play else -1
            node.value_sum += val*value
            node.visit_count+=1
    
    def rollout(self,node):
        """
        Perform a rollout
        """
        if self.args["random"] == 1:
            state = node.state
            if state.is_terminal():
                return state.reward()
            else:
                while not state.is_terminal():
                    player = state.Player_turn()
                    build,work = self.fast_board.convert_board_to_array(state)
                    pos_state = self.fast_board.all_possible_next_states(build,work,player)
                    next_state = pos_state[np.random.randint(0,len(pos_state))]
                    state = self.fast_board.convert_array_to_board(state,next_state[0],next_state[1])
                    if state.is_terminal():
                        return state.reward()   
        else:
            state = node.state
            if state.is_terminal():
                return state.reward()
            else:
                while not state.is_terminal():
                    player = state.Player_turn()
                    if player == "A":
                        ag = LinearRlAgentV2("A",depth=self.depth)
                        state = ag.action(state)
                    else:
                        ag = LinearRlAgentV2("B",depth=self.depth)
                        state = ag.action(state)
                    if state.is_terminal():
                        return state.reward()
    
    def run(self,to_play):
        """
        Perform One iteration of Selection,Expand,Rollout
        """
        if not self.root.is_expanded():
            self.root.expand()
        else:
            search_path = [self.root]
            current_node = self.root
            while current_node.is_expanded():
                current_node = self.root.select_child()
                search_path.append(current_node)
            for sim in range(self.args["Num_Simulations"]):
                reward = self.rollout(current_node)
                self.backpropagate(search_path,reward,current_node.state.Player_turn())
            if not current_node.state.is_terminal():
                current_node.expand()
        return self.root

    def breadth_run(self,node):
        """
        Performs a rollout for all child nodes regardless of UCB score
        """
        if not node.is_expanded():
            node.expand()
        if node.state.is_terminal():
            return node
        child_nodes = list(node.children)
        for child in child_nodes:
            search_path = [node,child]
            for sim in range(self.args["Num_Simulations"]):
                reward = self.rollout(child)
                self.backpropagate(search_path,reward,child.state.Player_turn())
        return node
                

    def collapse(self,node):
        """
        Return a list of all the nodes in the MCTS
        """
        all_nodes = set()
        to_explore = list(node.children.keys())
        all_nodes = all_nodes | set(node.children.keys())
        while to_explore != []:
            current = to_explore.pop()
            if current.children != {}:
                childs = list(current.children.keys())
                to_explore += childs
                all_nodes = all_nodes | set(childs)
            else:
                continue

        return all_nodes

    
class MCTS_Only_Agent(RandomAgent):
    def __init__(self,name,args):
        super().__init__(name)
        self.args = args
    
    def action(self,board):
        node = Node(board)
        mcts = MCTS(node,None,self.args)
        if board.is_terminal():
            return board
        if not node.is_expanded():
            node.expand()
        
        children = list(node.children)
        vals = [n.value() for n in children]
        for child in range(len(children)):
            for sim in range(self.args["Num_Simulations"]):
                reward = mcts.rollout(children[child])
                vals[child]+=reward
        if node.state.Player_turn() == "A":
            return children[np.argmax(vals)].state
        else:
            return children[np.argmin(vals)].state


    
def run_santorini(agent1 = LinearRlAgentV2("A"), agent2 = LinearRlAgentV2("B")):
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
            '''
            board_levels, worker_coords = FastBoard.convert_board_to_array(board)
            fast_board = FastBoard()
            start = time.time()
            print(MinimaxWithPruning(board_levels, worker_coords, current_player, 3, fast_board))
            end = time.time()
            print(f'tree with ab pruning took {end-start}')
            '''
            print(f'Current Player is {current_player}')
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


