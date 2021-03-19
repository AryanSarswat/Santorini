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
from runner import run_santorini

PATH = "Value_Func_State_Dict_CNN_B.pt"



class ValueFuncTrainer():
    def __init__(self, epochs, batches, nn, agent1, agent2):
        self.epochs = epochs
        self.batches = batches
        self.agent1 = agent1
        self.agent2 = agent2
        self.nn = nn
        self.state_values1 = []
        self.state_values2 = []
        self.batch_loss_array = []
        self.epoch_loss_array = []
    
    def train(self):
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0
            batch_counter = 0
            batch_loss = 0
            for batch in tqdm(range(self.batches)):
                winner = self.training_loop(self.agent1, self.agent2)
                #print(f"\n{winner}") 
                #print(f"\n{self.state_values}")
                with T.autograd.set_detect_anomaly(True):
                    while self.state_values1 != []:
                        self.nn.optimizer.zero_grad()
                        reward = self.agent1.reward(winner)
                        loss = self.nn.loss(self.state_values1.pop(-1), T.cuda.FloatTensor([[reward]])).to(self.nn.device)
                        loss.backward()
                        batch_loss += loss.item()
                        epoch_loss += batch_loss
                        reward = reward * 0.97
                    while self.state_values2 != []:
                        self.nn.optimizer.zero_grad()
                        reward = self.agent2.reward(winner)
                        loss = self.nn.loss(self.state_values2.pop(-1), T.cuda.FloatTensor([[reward]])).to(self.nn.device)
                        loss.backward()
                        batch_loss += loss.item()
                        epoch_loss += batch_loss
                        reward = reward * 0.97    
                    self.nn.optimizer.step()
                    self.nn.epsilon = self.nn.epsilon * 0.999 if self.nn.epsilon > self.nn.epsilon_min else self.nn.epsilon_min
                batch_counter += 1
                if batch_counter == 10:
                    self.batch_loss_array.append(batch_loss)
                    batch_loss = 0
                    batch_counter = 0
                #print(f"\n{batch_loss}")
            self.epoch_loss_array.append(epoch_loss)
            print(f"\n{epoch_loss}")    

    def training_loop(self, agent1, agent2):
        win = None
        board = Board(agent1, agent2)
        board = board.PlayerA.place_workers(board)
        board = board.PlayerB.place_workers(board)
        currentPlayer = board.PlayerA
        while win == None:
            win = board.start_turn_check_win(currentPlayer)
            if win != None:
                break
            if currentPlayer.name == self.agent1.name:
                states = board.all_possible_next_states(self.agent1.name)
                values = []
                rand = np.random.uniform()
                for state in states:
                    converted_state = self.agent1.convertTo2D(state)
                    values.append(self.nn.forward(converted_state).to(self.nn.device))
                if (self.agent1.explore == False) and (rand > self.nn.epsilon):
                    highest_value = T.argmax(T.cat(values)).item()
                    self.state_values1.append(values[highest_value])
                    board = states[highest_value]
                else:
                    choice = random.choice(values)
                    index = values.index(choice)
                    self.state_values1.append(choice)
                    board = states[index] 
            else:
                #board = currentPlayer.action(board)
                states = board.all_possible_next_states(self.agent2.name)
                values = []
                rand = np.random.uniform()
                for state in states:
                    converted_state = self.agent2.convertTo2D(state)
                    values.append(self.nn.forward(converted_state).to(self.nn.device))
                if (self.agent2.explore == False) and (rand > self.nn.epsilon):
                    highest_value = T.argmax(T.cat(values)).item()
                    self.state_values2.append(values[highest_value])
                    board = states[highest_value]
                else:
                    choice = random.choice(values)
                    index = values.index(choice)
                    self.state_values2.append(choice)
                    board = states[index] 
            win = board.end_turn_check_win(currentPlayer)
            if win != None:
                break
            if currentPlayer.name == board.PlayerA.name:
                currentPlayer = board.PlayerB
            else:
                currentPlayer = board.PlayerA
        return win

    def plot_loss_epoch(self):
        plt.plot(self.epoch_loss_array)
        plt.title("Loss versus epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_loss_batch(self):
        plt.plot(self.batch_loss_array)
        plt.title("Loss versus Batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.show()

def evaluate_model(brain):
    brain.nn.eval()
    count = 0
    total = 0
    with T.no_grad():
        for i in tqdm(range(1)):
            winner = run_santorini(brain, HumanPlayer("B"))
            if winner == brain.name:
                count += 1
                total += 1
            else:
                total += 1
    return count/total


if os.path.isfile(PATH):
    print("\n Loading Saved Model")
    brain1 = Agent("A", True)    
    #brain2 = Agent("B", False)
    #loaded_nn = ValueFunc()
    brain1.nn.load_state_dict(T.load(PATH))
    #loaded_nn.train()
    print(evaluate_model(brain1))
    #trainer = ValueFuncTrainer(100, 100, loaded_nn, brain1, brain2)
   #trainer.train()
    #T.save(trainer.nn.state_dict(),PATH)
   # trainer.plot_loss_epoch()
    #trainer.plot_loss_batch()

else:
    print("\n Training..........")
    brain = Agent("A", False)
    trainer = ValueFuncTrainer(50, 100, brain)
    trainer.train()
    T.save(trainer.nn.state_dict(),PATH)
    #print(trainer.agent.nn.epsilon)
    brain.plot_loss()
             
#print(T.cuda.is_available())
#print(os.environ.get('CUDA_PATH'))
#print(T.__version__)