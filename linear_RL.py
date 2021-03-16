from Game import *
import random, math
import numpy as np

#figure out out to organize classes in separate files later
class searchTree():
    '''
    Parameters:
    board - board object, acts as root node for search tree
    current_player - whether PlayerA or PlayerB is currently going to make a move
    depth 
    - how deep should the search tree go (each move = one lvl deeper regardless of player)
    - even number makes more sense, so that terminal nodes are at same player acting again)
    
    each searchTree contains a root_node, as well as all child node objects extending from it
    (basically more searchTree objects)

    '''
    def __init__(self, board, current_player, depth): 
        self.depth = depth
        self.root_node = board
        self.current_player = current_player
        #generate list of child nodes

    def find_child_nodes(self):
        # this function uses the find all possible moves to get all possible child nodes
        # recursively call searchTree(new_board, next player, and depth-1) and add to a list

class minimaxTree(searchTree):
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

class linearFnApproximator():
    def __init__(self, weights):
        #should board be in init?

    def calculate_state_value(self, board, name):
        '''
        input: game board object, player name, own weights
        output: numerical value of given game state
        utilizes weights + board state to calculate the state value
        '''


    def calculate_mobility_features(self, board, player, opponent):
        '''
        input: game board object, player name, opponent name
        output: numpy array with value of each mobility related feature
        list of features:
        1. possible moves that your workers can collectively make (normalize by )
        2. possible moves that your opponent's workers can collectively make
        3. number of moves that allow going from 0->1
        4. number of moves that allow going from 1->2
        5. number of moves that allow going from 2->3
        6,7,8. likewise for opponent.
        '''
        num_features = 8
        features = [0 for i in range(num_features)]
        player_possible_moves = board.all_possible_next_states(player)
        #feature 1 and 2
        max_possible_moves = 128
        features[0] = len(player_possible_moves)/max_possible_moves
        
    def num_moves_upward_from_state(self, board, player, level):
        




    def calculate_position_features(self, board):
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
        '''
        features = []
        center_x, center_y = 2,2
        player1 = board.PlayerA
        player2 = board.PlayerB

        #calculate features 1 - 6
        for player in [player1, player2]:
            for level in [0,1,2]:
                features.append(self.num_workers_on_level(board, player, level))
        
        #calculate features 7, 8
        def distance(x1, x2, y1, y2):
            return (x1-x2)^2 + (y1-y2)^2

        for player in [player1, player2]:
            total_dist = 0
            for worker in player.workers:
                worker_x, worker_y = worker.current_location[0], worker.current_location[1]
                total_dist += distance(worker_x, center_x, worker_y, center_y)
            features.append(total_dist)
        
        return features

    def num_workers_on_level(self, board, player, level):
        '''
        inputs: board, player, building level
        output: number of given player's workers on the given building level at that state
        '''
        output = 0
        for worker in player.workers:
            worker_level = worker.building_level
            if worker_level == level:
                output += 1
        return output

class linearRlAgent(RandomAgent):
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
    current_player = 'player_a'
    
    def get_current_board_player(current_player):
        if current_player == 'player_a':
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
            board.print_board()
            print("----------------------------------------------------------------\n")
            board = board_player.action(board)

            #because the board has been replaced, need to retrieve player obj again
            board_player = get_current_board_player(current_player)
            win = board.end_turn_check_win(board_player)
            if win != None:
                board.print_board()
                break
        
        if current_player == 'player_a':
            current_player = 'player_b'
        else:
            current_player = 'player_a'
        
    return win
    
run_santorini()