from Combined import run_santorini,LinearRlAgentV2,MCTS_Only_Agent,Trainer_CNN,RandomAgent
from MCTS_Trainer import MCTS_Agent


args = {
    'Num_Simulations': 1,
    'Iterations' : 10000,
    'Tree_depth' : 2,                     # Total number of MCTS simulations to run when deciding on a move to play
    'epochs': 1,
    'depth' : 25,                                    # Number of epochs of training per iteration
    'checkpoint_path': r"C:\Users\sarya\Documents\GitHub\Master-Procrastinator",
    'random' : 0
}

try:
    MCTS_ANN_Agent = MCTS_Agent("C")
except :
    print("ANN not Loaded")

try:
    MCTS_CNN_Agent = Trainer_CNN(args)
except :
    print("CNN not loaded")

Linear_A = LinearRlAgentV2("A")
Linear_B = LinearRlAgentV2("A")

Random_A = RandomAgent("A")
Random_B = RandomAgent("B")

MCTS_O_Agent = MCTS_Only_Agent("C",args)

Possible_Games = [(MCTS_ANN_Agent,MCTS_CNN_Agent),(MCTS_CNN_Agent,MCTS_ANN_Agent),(MCTS_ANN_Agent,Linear_B),(Linear_A,MCTS_ANN_Agent),(MCTS_CNN_Agent,Linear_B),(Linear_B,MCTS_CNN_Agent),
                (Linear_A,MCTS_O_Agent),(MCTS_O_Agent,Linear_B),(MCTS_ANN_Agent,MCTS_O_Agent),(MCTS_O_Agent,MCTS_ANN_Agent),(MCTS_CNN_Agent,MCTS_O_Agent),(MCTS_O_Agent,MCTS_CNN_Agent),
                 (Random_A,MCTS_ANN_Agent),(MCTS_ANN_Agent,Random_B),(Random_A,MCTS_CNN_Agent),(MCTS_CNN_Agent,Random_B), (MCTS_O_Agent,Random_B),(Random_A,MCTS_O_Agent)]

num_games = 50

dat = dict() 

for i in Possible_Games:
    wins = 0
    for j in range(num_games):
        if run_santorini(agent1=i[0],agent2=i[1]) == "A":
            wins+=1
    one = str(i).split()[0]
    two = str(i).split()[4]
    dat[(one,two)] = wins

for i in dat.items():
    print(i)
        