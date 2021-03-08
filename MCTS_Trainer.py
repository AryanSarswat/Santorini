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
        train_examples = []
        current_player = self.game.get_current_player()
        state = self.game

        while True:
            self.mcts = MCTS(self.game,self.model,self.args)
            root = self.mcts.run(self.model,state,current_player)

            state = root.select_child()
            current_player = state.get_current_player()

        pass

    def learn(self):
        for i in range(self.args['numIters']):
            train_examples = []
            for eps in range(self.args['numEps']):
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)
            
            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args["checkpoint_path"]
            self.save_checkpoint(folder=os.getcwd(),filename)

    def train(self):
        pass

    def loss_v(self,targets,outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/target.size([0])
        return loss

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