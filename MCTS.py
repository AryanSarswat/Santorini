from Game import *

class MCTS():
    """
    Class for containing the Monte Carlo Tree
    """

    def __init__(self,game,nnet):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Ns = {}
        