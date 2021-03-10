import numpy as np
import sys
from copy import deepcopy

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
        Overloaded constructor for training with 2 agents
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
        elif current_building_level < 3:
            possible_building_levels.append(current_building_level)
            possible_building_levels.append(current_building_level+1)
            possible_building_levels.append(current_building_level-1)

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
                    return "A wins!"
                else:
                    continue
        else:
            for worker in self.PlayerB.workers:
                if worker.building_level == 3:
                    return "B wins!"
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
                pos_moves.append(self.possible_worker_movements(location))
            if len(pos_moves) == 0:
                return "B wins!"
            else:
                return None
        else:
            pos_moves = []
            for worker in self.PlayerB.workers:
                location = worker.current_location
                pos_moves.append(self.possible_worker_movements(location))
            if len(pos_moves) == 0:
                return "A wins!"
            else:
                return None
    
    def is_terminal(self):
        """
        Returns True if the game is over
        Return False if game is not over
        """
        #Check if Player 1 Lost or Won
        if (self.start_turn_check_win(self.PlayerB) != None) or (self.end_turn_check_win(self.PlayerA) != None):
            PlayerAWin = True
        else:
            PlayerAWin = False
        #Check if Player 2 lost or won
        if (self.start_turn_check_win(self.PlayerA) != None) or (self.end_turn_check_win(self.PlayerB) != None):
            PlayerBWin = True
        else:
            PlayerBWin = False
        
        #Return False if both False
        if PlayerAWin == PlayerBWin:
            return False
        #Return True if somebody won
        else:
            return PlayerAWin if PlayerAWin else PlayerBWin
 
    def reward(self):
        """
        Return 1 if PlayerA won
        Returns -1 if PlayerB won
        Return 0 if game is not over
        """
        if self.is_terminal():
            if (self.start_turn_check_win(2) != None) or (self.end_turn_check_win(1) != None):
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
