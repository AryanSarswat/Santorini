import numpy as np
import sys
from Game import *
import random

class RandomAgent():
    '''
    RandomAgent class contains
    1) 2 distinguishable Worker pieces
    '''
    def __init__(self, name):
        self.name = name        
        self.workers = [Worker([], str(name)+"1"), Worker([], str(name)+"2")]

    
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
                board.board[coords[0]][coords[1]].update_worker(self.workers[place_count])
                place_count += 1
            except Exception:
                continue
        return board
    
    def action(self, board):
        """
        Method to randomly place a worker, afterwards, place a building
        """
        board = random.choice(board.all_possible_next_states(self.name))
        return board