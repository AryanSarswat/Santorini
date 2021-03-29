###Code for Running Game
from Game import *
import numpy as np
import time
from linear_rl_core_v2 import RandomAgent, LinearRlAgentV2, MinimaxWithPruning
from linear_rl_core_v1 import LinearRlAgentV1
from fast_board import FastBoard

def run_santorini(agent1 = RandomAgent("A"), agent2 = RandomAgent("B"), verbose = True, trainer_a = None, trainer_b = None):
    '''
    Runs a game of Santorini, allow choice of AI/human players
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
            if verbose:
                print(f'Current Player is {current_player}')
                board.print_board()
                print("----------------------------------------------------------------\n")
            if current_player == 'A':
                board = board_player.action(board, trainer_a)
            else:
                board = board_player.action(board, trainer_b)
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
    
#print(run_santorini(LinearRlAgentV2("A"), LinearRlAgentV1("B"), False))

