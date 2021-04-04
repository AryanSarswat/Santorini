###Code for Running Game
from Game import *
from linear_rl_agents import TreeStrapMinimax, RootStrapAB, LinearRlAgentV2, RandomAgent

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

#trained weights
rootstrap_depth3_self_play_100_games = [-1.70041383, -1.40308437,  3.81622973,  0.98649831,  0.18495751, -4.61974509, -1.57060762,  1.29561011]
treestrap_depth3_self_play_50_games = [-111.10484802, -105.02739914,  126.04215728,  128.71120153,   93.56648036, -133.40318024,  -52.95466135,   19.59279387]

#trainer objects
rootstrap = RootStrapAB()
treestrap = TreeStrapMinimax([-57.1350499,  -24.43606518, 87.43759999,  70.55689126,  61.53952637, -48.80110254, -13.22514194,  29.42421974])

#initialize agents
agent_a = LinearRlAgentV2('A', 2)
agent_b = LinearRlAgentV2('B', 2, rootstrap_depth3_self_play_100_games)

'''
Test Results: (at Depth 3, 100 games per side)
- Treestrap as Player A vs Manual as Player B (53 vs 47)
- Manual as Player A vs Treestrap as Player B (52 vs 48)
- Rootstrap as Player A vs Manual as Player B (62 vs 38)
- Manual as Player A vs Rootstrap as Player B (54 vs 46)
'''

#mention how trained weighst are optimized for their specific search depth

if __name__=='__main__':
    run_santorini(LinearRlAgentV2("A", 5, treestrap_depth3_self_play_50_games), HumanPlayer("B"))
    #training_loop(rootstrap, treestrap, agent_a, agent_b, 1)

