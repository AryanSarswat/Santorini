import numpy as np
import sys

def get_neighbours(coord):
    '''
    Returns a list of all neighbour location from a given state
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

class Square():
    """
    Element used to represent the a square on the board
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
            self.building_level+=1
        else:
            raise Exception("Max Building Error Reached")
    
    def update_worker(self,Worker):
        #Update Square if worker moves on the square
        if self.worker == None:
            self.worker = Worker
        else:
            raise Exception("Worker Already Present in this Square")


class Board():
    '''
    Board class contains
    1) The entire board which is a list of Squares
    2) List of all workers for both players
    3) A list of all of Player 1's workers
    3) A list of all of Player 2's workers
    '''
    def __init__(self):
        self.board = []
        self.workers = []
        self.Player_1_Workers = []
        self.Player_2_Workers = []
        for i in range(5):
            temp_board = []
            for j in range(5):
                temp_board.append(Square(i,j))
            self.board.append(temp_board) 
    
    
    def intialize_workers(self,Player_1_Worker_Locations,Player_2_Worker_Locations):
        '''
        Intializes the workers in the board
        Player_n_Location is a list of coordinates [[row,col],[row,col]] where the workers are placed
        '''
        #Ensure all locations are unique
        all_locations = Player_1_Worker_Locations + Player_2_Worker_Locations
        if len(np.unique(all_locations,axis=0)) != len(all_locations):
            raise Exception("Workers Cannot have the same locations")
        
        #Ensure all locations are within the board
        for coord in all_locations:
            if (coord[0] > 4 or coord [0] < 0) or (coord[1] > 4 or coord[1] < 0):
                raise Exception("Incorrect Co-ordinates Entered please enter coordinates within the 5 X 5 board")
            else:
                continue
    
        #Create a list of all workers:
        self.Player_1_Workers = [Worker(Player_1_Worker_Locations[0],"W11"),Worker(Player_1_Worker_Locations[1],"W12")]
        self.Player_2_Workers = [Worker(Player_2_Worker_Locations[0],"W21"),Worker(Player_2_Worker_Locations[1],"W22")]
        self.workers += self.Player_1_Workers + self.Player_2_Workers
        #Intialize Workers into the board
        for worker in self.workers:
            row,col = worker.current_location
            self.board[row][col].update_worker(worker)
  
    
    def print_board(self):
        '''
        Prints the board into a readable format
        '''
        for row in self.board:
            temp_list = []
            for square in row:
                if square.worker != None :
                    temp_list.append((square.building_level,square.worker.name))
                else:
                    temp_list.append((square.building_level,square.worker))
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
        '''
        #Get Worker
        worker = self.board[previous_location[0]][previous_location[1]].worker
        #Update Worker
        worker.current_location = new_location
        worker.previous_location = previous_location
        worker.building_level = self.board[new_location[0]][new_location[1]].building_level
        #Update Location
        self.board[new_location[0]][new_location[1]].update_worker(worker)
        self.board[worker.previous_location[0]][worker.previous_location[1]].worker = None
        pass
    
    #Build a building
    def update_building_level(self,coord):
        '''
        Functionality to add a building level
        '''
        self.board[coord[0]][coord[1]].add_building()
        pass


    
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
    
    def end_turn_check_win(self,Player):
        '''
        Checks if a player's workers have reached level 3
        '''
        if Player == 1:
            for worker in self.Player_1_Workers:
                coord = worker.current_location
                if self.board[coord[0]][coord[1]].building_level == 3:
                    return "1"
                else:
                    continue
        else:
            for worker in self.Player_2_Workers:
                coord = worker.current_location
                if self.board[coord[0]][coord[1]].building_level == 3:
                    return "2"
                else:
                    continue
        return None
    
    def start_turn_check_win(self,Player,board):
        '''
        Check at the start of the turn to see if other player has won due to no possible worker movements
        '''
        if Player == 1:
            pos_moves = []
            for worker in self.Player_1_Workers:
                location = worker.current_location
                pos_moves.append(board.possible_worker_movements(location))
            if pos_moves == []:
                return "2"
            else:
                return None
        else:
            pos_moves = []
            for worker in self.Player_2_Workers:
                location = worker.current_location
                pos_moves.append(board.possible_worker_movements(location))
            if pos_moves == []:
                return "1"
            else:
                return None
