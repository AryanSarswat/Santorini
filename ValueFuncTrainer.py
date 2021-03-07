from ValueFuncAI import *
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Game import *

def training_loop(agent1, agent2):
    win = None
    board = Board(agent1, agent2)
    board = board.PlayerA.place_workers(board)
    board = board.PlayerB.place_workers(board)
    currentPlayer = board.PlayerA
    while win == None:
        win = board.start_turn_check_win(currentPlayer)
        if win != None:
            break
        else:
            #board.print_board()
            #print("----------------------------------------------------------------\n")
            board = currentPlayer.action(board)
            win = board.end_turn_check_win(currentPlayer)
            if win != None:
                #board.print_board()
                break

        if currentPlayer == board.PlayerA:
            currentPlayer = board.PlayerB
        else:
            currentPlayer = board.PlayerA
    return win

class ValueFuncTrainer():
    def __init__(self, epochs, batches, agent):
        self.epochs = epochs
        self.batches = batches
        self.agent = agent

    
    def train(self):
        for epoch in tqdm(self.epochs):
            for batch in self.batches:
                winner = training_loop(self.agent, RandomAgent())
                while self.agent.values != []:
                    self.agent.nn.optimizer.zero_grad()
                    reward = self.agent.reward(winner)
                    loss = self.agent.nn.loss(self.agent.values.pop(-1), reward)
                    loss.backwards()
                    self.agent.nn.optimizer.step()
                    reward = reward * 0.98
                self.agent.nn.epsilon = self.agent.nn.epsilon * 0.99


    def save_checkpoint(self,folder,filename):
        """
        Save the Neural Network
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        filepath = os.path.join(folder,filename)
        T.save({
            'state_dict' : self.agent.nn.state_dict(),
        },filepath) 

trainer = ValueFuncTrainer(5, 5, Agent())
#trainer.save_checkpoint("C:Santorini", "model1")  
             
