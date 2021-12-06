import numpy as np
from Game import Board
from copy import deepcopy

class FastBoard():
    '''
    collection of attributes and methods for quickly retrieving valid moves.
    faster implementation of the game

    workflow: 
    1. convert board to np array format
    2. search thru moves and calculate minimax values 
    3. figure out 'best' move given a certain depth 
    4. convert 'best' move back to regular board obj to update game state.

    board representation:
    - use 5x5 np array for building levels
    - 4x2 np array for worker coords. in order of A1, A2, B1, B2
    - current player variable
    '''

    #assumes 5x5 board, pre-calculate valid_coordinates given a square
    #key: board coord
    #value: list containing valid board coordinates. prioritize moves going toward center.

    def __init__(self):
        self.create_neighbour_coords_dictionary()

    def create_neighbour_coords_dictionary(self):
        self.valid_coord_dict = dict()
        for row_num in range(5):
            for col_num in range(5):
                coord = (row_num, col_num)
                self.valid_coord_dict[coord] = FastBoard.get_neighbours(coord)

    @staticmethod
    def convert_board_to_array(board):
        '''
        takes in board object, converts to representation using np arrays
        outputs: 5x5 array containing square levels
        2x4x1 array containing worker positions (A1, A2, B1, B2)
        '''
        board_levels = []
        for row in board.board:
            board_row = []
            for square in row:
                board_row.append(square.building_level)
            board_levels.append(board_row)
        worker_coords = []
        for worker in board.PlayerA.workers:
            worker_coords.append(tuple(worker.current_location))
        for worker in board.PlayerB.workers:
            worker_coords.append(tuple(worker.current_location))
        board_levels = np.array(board_levels)
        #worker_coords = np.array(worker_coords)
        return board_levels, worker_coords

    def retrieve_valid_worker_moves(self, board_levels, all_worker_coords, worker_coords):
        '''
        Inputs: 
        - board_levels, coords of all workers, and worker_coords of specific worker in tuple format (x,y)

        Outputs:
        - list of all valid worker moves from current state
        '''
        valid_squares = self.valid_coord_dict[worker_coords]
        current_building_level = board_levels[worker_coords[0]][worker_coords[1]]
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
        #Filter based on Building level
        possible_movements = list(filter(lambda element : board_levels[element[0]][element[1]] in possible_building_levels, valid_squares))
        #Filter based on Occupancy
        possible_movements  = list(filter(lambda element: (element[0],element[1]) not in all_worker_coords, possible_movements))
        return possible_movements
        
    def valid_building_options(self, board_levels, all_worker_coords, location):
        '''
        Returns all the possible building options at a given location. 
        Uses all_worker_coords from after the worker has been moved.
        '''
        neighbour_locations = self.valid_coord_dict[location]
        possible_options = []
        for row,col in neighbour_locations:
                if board_levels[row][col] < 4  and ((row, col) not in all_worker_coords):
                    possible_options.append([row,col])
        return possible_options

    def all_possible_next_states(self, board_levels, all_worker_coords, current_player):
        """
        Return a copy of all the possible next states after a worker movement and a bulding placement
        Output Format: List containing tuples with (board_levels, all_worker_coords)
        """
        if current_player == 'A':
            worker_index = [0,1]
        else:
            worker_index = [2,3]
        possible_worker_1_movements = self.retrieve_valid_worker_moves(board_levels, all_worker_coords, all_worker_coords[worker_index[0]])
        possible_worker_2_movements = self.retrieve_valid_worker_moves(board_levels, all_worker_coords, all_worker_coords[worker_index[1]])
        possible_next_state_1 = []
        possible_next_state_2 = []

        #creates list of possible worker coord states
        for action in possible_worker_1_movements:
            worker_coords = all_worker_coords.copy()
            worker_coords[worker_index[0]] = action
            possible_next_state_1.append(worker_coords)
        for action in possible_worker_2_movements:
            worker_coords = all_worker_coords.copy()
            worker_coords[worker_index[1]] = action
            possible_next_state_2.append(worker_coords)
     
        #Return the possible build options
        possible_builds_1 = [self.valid_building_options(board_levels, all_worker_coords, all_worker_coords[worker_index[0]]) for all_worker_coords in possible_next_state_1]
        possible_builds_2 = [self.valid_building_options(board_levels, all_worker_coords, all_worker_coords[worker_index[1]]) for all_worker_coords in possible_next_state_2]
        next_states = []
        #Apply build and return a deepcopy of the new state
        for n in range(len(possible_builds_1)):
            builds = possible_builds_1[n]
            altered_worker_coords = possible_next_state_1[n]
            for b in builds:
                altered_board_level = board_levels.copy()
                altered_board_level[b[0]][b[1]] += 1
                next_states.append((altered_board_level, altered_worker_coords))                    
        for n in range(len(possible_builds_2)):
            builds = possible_builds_2[n]
            altered_worker_coords = possible_next_state_2[n]
            for b in builds:
                altered_board_level = board_levels.copy()
                altered_board_level[b[0]][b[1]] += 1
                next_states.append((altered_board_level, altered_worker_coords))   

                
        def getHeight(state):
            '''
            function that retrieves the sum of squares of player's worker heights
            allows for move ordering so that better moves are prioritized in minimax search

            squared worker height is used to prioritize moves to level 3>2>1.
            '''
            board_level, worker_coords = state
            total_worker_height = 0
            for index in worker_index:
                worker_pos = worker_coords[index]
                total_worker_height += (board_level[worker_pos[0]][worker_pos[1]])**2
            return total_worker_height
        next_states.sort(reverse=True, key = getHeight)
        return next_states

    @staticmethod
    def get_neighbours(coord):
        '''
        Returns a list of all neighbour location from a given state. Assumes 5x5 board.
        '''
        neighbour_locations = [(coord[0]+1,coord[1]),(coord[0],coord[1]+1),(coord[0]+1,coord[1]+1),(coord[0]-1,coord[1]),(coord[0],coord[1]-1),(coord[0]-1,coord[1]-1),(coord[0]+1,coord[1]-1),(coord[0]-1,coord[1]+1)]
        #Only keep coordinates within the board
        neighbour_locations = tuple(filter(lambda element: (element[0] >= 0 and element[0] < 5) and (element[1] >= 0 and element[1] < 5),neighbour_locations))
        return neighbour_locations

    @staticmethod
    def convert_array_to_board(old_board_obj, board_levels, worker_coords):
        board = deepcopy(old_board_obj)
        num_rows, num_cols = 5,5
        #update square building levels and worker
        total_building_count = 0
        for row in range(num_rows):
            for col in range(num_cols):
                board.board[row][col].building_level = board_levels[row][col]
                total_building_count += board_levels[row][col]
                if (row,col) in worker_coords:
                    worker_index = worker_coords.index((row,col))
                    if worker_index == 0 or worker_index == 1:
                        board.board[row][col].worker = board.PlayerA.workers[worker_index]
                    else:
                        board.board[row][col].worker = board.PlayerB.workers[worker_index-2]
                else:
                    board.board[row][col].remove_worker()
        #update total building count
        board.total_building_count = total_building_count

        a_worker_coords = worker_coords[:2]
        b_worker_coords = worker_coords[2:]
        #update worker objects
        for worker_num in range(len(a_worker_coords)):
            worker_coords = a_worker_coords[worker_num]
            board.PlayerA.workers[worker_num].update_location(list(worker_coords))
            new_level = board_levels[worker_coords[0]][worker_coords[1]]
            board.PlayerA.workers[worker_num].update_building_level(new_level)
        for worker_num in range(len(b_worker_coords)):
            worker_coords = b_worker_coords[worker_num]
            board.PlayerB.workers[worker_num].update_location(list(worker_coords))
            new_level = board_levels[worker_coords[0]][worker_coords[1]]
            board.PlayerB.workers[worker_num].update_building_level(new_level)
        
        return board
#move ordering: want to examine upward moves first, followed by central moves