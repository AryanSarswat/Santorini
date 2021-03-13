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
from RandomAgent import *

PATH = "Value_Func_State_Dict_CNN.pt"

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
            board = currentPlayer.action(board)
            win = board.end_turn_check_win(currentPlayer)
            if win != None:
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
        count = 0
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0
            for batch in range(self.batches):
                winner = training_loop(self.agent, RandomAgent("B"))
                while self.agent.state_values != []:
                    self.agent.nn.optimizer.zero_grad()
                    reward = self.agent.reward(winner)
                    loss = self.agent.nn.loss(self.agent.state_values.pop(-1), T.Tensor([[reward]]))
                    loss.backward()
                    epoch_loss += loss.item()
                    self.agent.nn.optimizer.step()
                    reward = reward * 0.98
                self.agent.nn.epsilon = self.agent.nn.epsilon * 0.99 if self.agent.nn.epsilon > self.agent.nn.epsilon_min else self.agent.nn.epsilon_min
                count+=1
                print("Count = " + str(count))
                
            self.agent.loss_array.append(epoch_loss)

    

if os.path.isfile(PATH):
    print("\n Loading Saved Model")
    brain = Agent(False)
    brain.nn.load_state_dict(T.load(PATH))
    count = 0
    total = 0
    for i in range(5):
        winner = training_loop(brain, RandomAgent("B"))
        if winner == "A":
            count += 1
            total += 1
        else:
            total += 1
    print(count/total)
    #trainer = ValueFuncTrainer(10, 10, brain)

else:
    print("\n Training..........")
    brain = Agent(True)
    trainer = ValueFuncTrainer(2, 10, brain)
    trainer.train()
    T.save(brain.nn.state_dict(),PATH)
    brain.plot_loss()
             
