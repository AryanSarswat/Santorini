###Code for Running Game
from Game import *
from linear_rl_agents import TreeStrapMinimax, RootStrapAB, LinearRlAgentV2, LinearRlAgentV3, RandomAgent
from linear_rl_core_v2 import LinearFnApproximatorV2
from fast_board import FastBoard

def run_santorini(agent1, agent2, verbose = True, trainer_a = None, trainer_b = None):
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
                # board_levels, all_worker_coords = FastBoard.convert_board_to_array(board)
                # fast_board = FastBoard()                
                # print(LinearFnApproximatorV2(board_levels, all_worker_coords, [1 for i in range(22)], fast_board))
                print(f'Current Player is {current_player}')
                board.print_board()
                print("----------------------------------------------------------------\n")
            if trainer_a == None and trainer_b == None: #i.e. not in training mode
                if current_player == 'A':
                    board = board_player.action(board)
                else:
                    board = board_player.action(board)
            else: #in training mode, only works for linearRLAgents which accept trainers in actions
                if current_player == 'A':
                    board = board_player.action(board, trainer_a)
                else:
                    board = board_player.action(board, trainer_b)
            #because the board has been replaced, need to retrieve player obj again
            board_player = get_current_board_player(current_player)
            win = board.end_turn_check_win(board_player)
            if win != None:
                if verbose: board.print_board()
                break
        
        if current_player == 'A':
            current_player = 'B'
        else:
            current_player = 'A'
        
    return win

def training_loop(trainer_a, trainer_b, agent_a, agent_b, n_iterations):
    '''
    executes training loop for linear agents to learn weights for linear approximator
    Inputs: 
    - either treestrap or rootstrap object for trainer_a and trainer_b
    - linear_rl_agents for agent_a and agent_b
    - number of iterations
    Outputs:
    prints out weights after every game trained, remember to save them manually for now
    '''
    a_wins = 0
    b_wins = 0
    for i in range(n_iterations):
        win = run_santorini(agent_a, agent_b, False, trainer_a, trainer_b)
        if win == 'A':
            a_wins += 1
        elif win == 'B':
            b_wins += 1
        print(f'Trainer A: {trainer_a}, Trainer B: {trainer_b}')
        print(f'{i+1}/{n_iterations} games completed. A has won {a_wins}/{i+1} games while B has won {b_wins}/{i+1} games.')
    #np.savetxt("rootstrap_weights.csv", trainer_a.weights, delimiter=",")

#trained weights
rootstrap_depth3_self_play_100_games = [-1.70041383, -1.40308437,  3.81622973,  0.98649831,  0.18495751, -4.61974509, -1.57060762,  1.29561011]
treestrap_depth3_self_play_50_games = [-111.10484802, -105.02739914,  126.04215728,  128.71120153,   93.56648036, -133.40318024,  -52.95466135,   19.59279387]
#rootstrap_test = np.loadtxt('rootstrap_weights.csv', delimiter = ',')
treestrap_test = [-141.89040637, -114.82240201,  355.79161609,  -37.17284442,   87.02059488, 27.31294783,  211.34971125,  170.23468898, -165.93052754,   36.91466959,-64.8510126,   -11.38154015 ,  56.84910486,  125.10571383,  134.26773231,-30.56365529,   -5.11376878,   45.41138929 ,  52.24639551,   68.9481104, 10.16104757,  -29.2034215]

#trainer objects
rootstrap = RootStrapAB()
treestrap = TreeStrapMinimax()

#initialize agents
agent_a = LinearRlAgentV2('A', 3, adaptive_search=True)
agent_b = LinearRlAgentV2('B', 3)

'''
Test Results: (at Depth 3, 100 games per side)
- Treestrap as Player A vs Manual as Player B (53 vs 47)
- Manual as Player A vs Treestrap as Player B (52 vs 48)
- Rootstrap as Player A vs Manual as Player B (62 vs 38)
- Manual as Player A vs Rootstrap as Player B (54 vs 46)
'''


if __name__=='__main__':
    #run_santorini(agent_a, agent_b)
    training_loop(None, None, agent_a, agent_b, 100)
    #a = np.loadtxt('trained_weights.csv', delimiter = ',')
    #print(a)    

#mention how trained weighst are optimized for their specific search depth
#adaptive search depth
#for report, find a reference to cite for minimax/alpha beta pruning

#training times for new v2approximator
#3 hours - 25 games of treestrap at depth 3
#2.5 hours - 200 games of rootstrap at depth 3
#performance still poor after this though, possibly due to increased complexity of approximator