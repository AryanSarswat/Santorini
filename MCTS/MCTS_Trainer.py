import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim

from MCTS import MCTS

class Trainer:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)

    def execute_episode(self):
        pass

    def learn(self):
        pass

    def train(self):
        pass

    def loss_v(self,targets,outputs):
        pass

    def save_checkpoint(self,folder,filename):
        """
        Save the Neural Network
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        filepath = os.path.join(folder,filename)
        torch.save({
            'state_dict' : self.model.state_dict(),
        },filepath)