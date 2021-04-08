from Combined import MCTS_Only_Agent,Trainer_CNN,RandomAgent,Neural_Network,ValueFunc
from MCTS_Trainer import MCTS_Agent
from Game import Board,HumanPlayer
import torch
from tqdm import tqdm
from GA import rootstrap_depth3_self_play_100_games
from linear_rl_agents import LinearRlAgentV3,LinearRlAgentV2

def run_santorini(agent1 = LinearRlAgentV2("A",4), agent2 = LinearRlAgentV2("B",4)):
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
            '''
            board_levels, worker_coords = FastBoard.convert_board_to_array(board)
            fast_board = FastBoard()
            start = time.time()
            print(MinimaxWithPruning(board_levels, worker_coords, current_player, 3, fast_board))
            end = time.time()
            print(f'tree with ab pruning took {end-start}')
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




args = {
    'Num_Simulations': 1,
    'Iterations' : 10000,
    'Tree_depth' : 2,                     # Total number of MCTS simulations to run when deciding on a move to play
    'epochs': 1,
    'depth' : 25,                                    # Number of epochs of training per iteration
    'checkpoint_path': r"C:\Users\sarya\Documents\GitHub\Master-Procrastinator",
    'random' : 0
}

model_ANN = Neural_Network()
model_ANN.load_state_dict(torch.load(r"C:\Users\sarya\Documents\GitHub\Master-Procrastinator\MCTS_AI_ANN"))
model_ANN.eval()

model_CNN = ValueFunc()
model_CNN.load_state_dict(torch.load(r"C:\Users\sarya\Documents\GitHub\Master-Procrastinator\MCTS_AI_CNN"))
model_CNN.eval()

#try:
MCTS_ANN_Agent_A = MCTS_Agent("A",NN=model_ANN)
MCTS_ANN_Agent_B = MCTS_Agent("B",NN=model_ANN)
#except :
    #print("ANN not Loaded")

#try:
MCTS_CNN_Agent_A = Trainer_CNN("A",args,NN=model_CNN)
MCTS_CNN_Agent_B = Trainer_CNN("B",args,NN=model_CNN)
#except :
    #print("CNN not loaded")

Linear_A = LinearRlAgentV2("A",5,trained_weights = rootstrap_depth3_self_play_100_games)
Linear_B = LinearRlAgentV2("B",4,trained_weights = rootstrap_depth3_self_play_100_games)

Random_A = RandomAgent("A")
Random_B = RandomAgent("B")

MCTS_O_Agent_A = MCTS_Only_Agent("A",args)
MCTS_O_Agent_B = MCTS_Only_Agent("B",args)

"""
Possible_Games = [(Linear_A,MCTS_O_Agent_A),(MCTS_O_Agent_A,Linear_B),
                    (Random_A,MCTS_O_Agent_B),(MCTS_O_Agent_A,Random_B)]

num_games = 50

dat = dict() 

for i in Possible_Games:
    wins = 0
    for j in tqdm(range(num_games)):
        if run_santorini(agent1=i[0],agent2=i[1]) == "A":
            wins+=1
    one = str(i).split()[0]
    two = str(i).split()[4]
    dat[(one,two)] = wins
for i in dat.items():
    print(i)
"""

run_santorini(agent1=Linear_A,agent2=HumanPlayer("B"))