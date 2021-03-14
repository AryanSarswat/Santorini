from Game import *
from MCTS import MCTS
from MCTS_Trainer import MCTS_Agent,Trainer
from MCTS_NN import Neural_Network

args = {
    'Num_Simulations': 10,                     # Total number of MCTS simulations to run when deciding on a move to play
    'epochs': 1,
    'depth' : 2,                                    # Number of epochs of training per iteration
    'checkpoint_path': r"C:\Users\sarya\Documents\GitHub\Master-Procrastinator"
}

AI = Trainer(args)
AI.initialize_mcts()
AI.train()
