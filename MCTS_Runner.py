from Game import *
from MCTS import MCTS
from MCTS_Trainer import Agent
from MCTS_NN import Neural_Network
A = HumanPlayer("A")
B = HumanPlayer("B")

board = Board(A,B)

board.PlayerA.place_workers(board)
board.PlayerB.place_workers(board)

board.print_board()

args = {
    'numIters': 10,                                # Total number of training iterations
    'Num_Simulations': 100,                     # Total number of MCTS simulations to run when deciding on a move to play
    'epochs': 2,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'
}

AI = Agent(board,Neural_Network(),args)

AI.train()
