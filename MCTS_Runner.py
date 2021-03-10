from Game import *
from MCTS import MCTS


A = HumanPlayer("A")
B = HumanPlayer("B")

board = Board(A,B)

board.PlayerA.place_workers(board)
board.PlayerB.place_workers(board)

board.print_board()

args = {
    'batch_size': 64,
    'numIters': 5,                                # Total number of training iterations
    'Num_Simulations': 100,                     # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 5,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 2,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'
}

mcts = MCTS(board,"Model",args)

root = mcts.run("Model",board,board.Player_turn())