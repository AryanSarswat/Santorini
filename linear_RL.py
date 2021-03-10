from Game import *
import random

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

'''
class linearFnApproximator():
    def __init__(self):
        self.feature_values
        self.weights = 
        self.state_values
    def calculate_features(self, state):
        
        takes in a given board state and returns feature values.
        '''

###Code for Running Game
def run_santorini(agent1 = HumanPlayer("A"), agent2 = RandomAgent("B")):
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