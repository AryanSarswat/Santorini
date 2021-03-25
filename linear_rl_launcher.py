###Code for Running Game
from Game import *
import numpy as np
import time
from linear_rl_core import RandomAgent, LinearRlAgent, MinimaxWithPruning, SearchTree, MinimaxTree
from fast_board import FastBoard

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
            #test_board = FastBoard()
            #board_levels, worker_coords = FastBoard.convert_board_to_array(board)
            #print(test_board.all_possible_next_states(board_levels, worker_coords, current_player))
            '''
            start = time.time()
            #print(MinimaxTree(board, current_player, depth=3))
            end = time.time()
            print(f'normal minimax tree took {end-start}')

            start = time.time()
            print(MinimaxWithPruning(board, current_player, depth=3))
            end = time.time()
            print(f'tree with ab pruning took {end-start}')
            #print(LinearFnApproximator(board))
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
    
#run_santorini()


import time
import copy
board = Board(agent1 = RandomAgent("A"), agent2 = RandomAgent("B"))
board = board.PlayerA.place_workers(board)
board = board.PlayerB.place_workers(board)
start = time.time()
for i in range(1):
    print(len(board.all_possible_next_states(board.PlayerA.name)))
end = time.time()
print(end-start)

test_board = FastBoard()
board_levels, worker_coords = FastBoard.convert_board_to_array(board)
start = time.time()
for i in range(1):
    print(len(test_board.all_possible_next_states(board_levels, worker_coords, 'A')))
end = time.time()
print(end-start)

# board = [((2, None), (0, None), (0, 'B2'), (0, None), (0, None)),
# ((0, None), (4, None), (1, None), (0, None), (0, None)),
# ((0, None), (0, None), (3, None), (0, 'A1'), (0, None)),
# ((0, None), (0, None), (0, None), (1, None), (0, 'B1')),
# ((0, 'A2'), (0, None), (0, None), (0, None), (0, None))]

# start = time.time()
# for i in range(10000):
#     board2 = board.copy()
#     board2[0] = ((3, None), (0, None), (0, 'B2'), (0, None), (0, None))
#     board2[1] = ((0, None), (4, 'B2'), (1, None), (0, None), (0, None))
# end = time.time()
# print(end-start)

# board = [[(2, None), (0, None), (0, 'B2'), (0, None), (0, None)],
# [(0, None), (4, None), (1, None), (0, None), (0, None)],
# [(0, None), (0, None), (3, None), (0, 'A1'), (0, None)],
# [(0, None), (0, None), (0, None), (1, None), (0, 'B1')],
# [(0, 'A2'), (0, None), (0, None), (0, None), (0, None)]]

# start = time.time()
# for i in range(10000):
#     board2 = np.array(board)
#     board3 = np.copy(board2)
# end = time.time()
# print(end-start)

