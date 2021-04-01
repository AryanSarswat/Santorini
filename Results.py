from Combined import run_santorini,LinearRlAgentV2,MCTS_Only_Agent,Trainer_CNN,RandomAgent,Neural_Network,ValueFunc
from MCTS_Trainer import MCTS_Agent
import torch

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

try:
    MCTS_ANN_Agent_A = MCTS_Agent("A",NN=model_ANN)
    MCTS_ANN_Agent_B = MCTS_Agent("B")
except :
    print("ANN not Loaded")

try:
    MCTS_CNN_Agent_A = Trainer_CNN("A",args,NN=model_CNN)
    MCTS_CNN_Agent_B = Trainer_CNN("B",args)
except :
    print("CNN not loaded")

Linear_A = LinearRlAgentV2("A")
Linear_B = LinearRlAgentV2("B")

Random_A = RandomAgent("A")
Random_B = RandomAgent("B")

MCTS_O_Agent_A = MCTS_Only_Agent("A",args)
MCTS_O_Agent_B = MCTS_Only_Agent("B",args)


Possible_Games = [(MCTS_ANN_Agent_A,MCTS_CNN_Agent_B),(MCTS_ANN_Agent_A,Linear_B),(MCTS_ANN_Agent_A,Random_B),(MCTS_ANN_Agent_A,MCTS_O_Agent_B),
                    (MCTS_CNN_Agent_A,MCTS_ANN_Agent_B),(Linear_A,MCTS_ANN_Agent_B),(Random_A,MCTS_ANN_Agent_B),(MCTS_O_Agent_A,MCTS_ANN_Agent_B),
                    (MCTS_CNN_Agent_A,Linear_B),(MCTS_CNN_Agent_A,Random_B),(MCTS_CNN_Agent_A,MCTS_O_Agent_B),
                    (Linear_A,MCTS_CNN_Agent_B),(Random_A,MCTS_CNN_Agent_B),(MCTS_O_Agent_A,MCTS_CNN_Agent_B),
                    (Linear_A,MCTS_O_Agent_A),(MCTS_O_Agent_A,Linear_B),
                    (Random_A,MCTS_O_Agent_B),(MCTS_O_Agent_A,Random_B)]

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