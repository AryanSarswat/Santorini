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

PATH = "Value_Func_State_Dict_CNN_random2.pt"
PATH2 = "Value_Func_State_Dict_CNN_D.pt"



class ValueFuncTrainer():
    def __init__(self, epochs, batches, nn, agent1, agent2):
        self.epochs = epochs
        self.batches = batches
        self.agent1 = agent1
        self.agent2 = agent2
        self.nn = nn
        self.state_values1 = []
        self.iter_loss_array = []
        self.epoch_loss_array = []
    
    def train(self):
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0
            for batch in tqdm(range(self.batches)):
                iter_loss = 0
                winner = self.training_loop(self.agent1, self.agent2)
                #print(f"\n{winner}") 
                #print(f"\n{self.state_values}")
                reward = self.agent1.reward(winner)
                with T.autograd.set_detect_anomaly(True):
                    while self.state_values1 != []:
                        self.nn.optimizer.zero_grad()
                        loss = self.nn.loss(self.state_values1.pop(-1), T.cuda.FloatTensor([[reward]])).to(self.nn.device)
                        loss.backward()
                        item = loss.item()
                        iter_loss += item
                        #print(f"\n{item}")
                        epoch_loss +=item
                        reward = reward * 0.96   
                    self.nn.optimizer.step()
                    self.nn.epsilon = self.nn.epsilon * 0.999 if self.nn.epsilon > self.nn.epsilon_min else self.nn.epsilon_min
                    self.iter_loss_array.append(iter_loss)                
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
                board = currentPlayer.action(board)
                '''states = board.all_possible_next_states(self.agent2.name)
                values = []
                rand = np.random.uniform()
                for state in states:
                    converted_state = self.agent2.convertTo2D(state)
                    values.append(self.nn.forward(converted_state).to(self.nn.device))
                if (self.agent2.explore == False) and (rand > self.nn.epsilon):
                    highest_value = T.argmax(T.cat(values)).item()
                    board = states[highest_value]
                else:
                    choice = random.choice(values)
                    index = values.index(choice)
                    board = states[index] '''
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

    def plot_loss_iter(self):
        plt.plot(self.iter_loss_array)
        plt.title("Loss versus Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()


    

def evaluate_model(brain1, brain2):
    brain2.nn.eval()
    #brain2.nn.eval()
    count = 0
    total = 0
    with T.no_grad():
        for i in tqdm(range(100)):
            winner = run_santorini(brain1, brain2)
            if winner == brain2.name:
                count += 1
                total += 1
            else:
                total += 1
    return count/total


if os.path.isfile(PATH):
    print("\n Loading Saved Model")
    brain1 = Agent("B", False)    
    #brain2 = Agent("B", False)
    #loaded_nn = ValueFunc()
    brain1.nn.load_state_dict(T.load(PATH2))
    print(evaluate_model(RandomAgent("A"), brain1))
    #trainer = ValueFuncTrainer(1000, 10, loaded_nn, brain1, RandomAgent("B"))
    #trainer.train()
    #T.save(trainer.nn.state_dict(),PATH2)
    #trainer.plot_loss_epoch()

else:
    print("\n Training..........")
    brain = Agent("A", False)
    loaded_nn = ValueFunc()
    loaded_nn.load_state_dict(T.load(PATH2))
    trainer = ValueFuncTrainer(100, 100, loaded_nn, brain, RandomAgent("B"))
    trainer.train()
    T.save(trainer.nn.state_dict(),PATH)
    #print(trainer.agent.nn.epsilon)
    trainer.plot_loss_iter()
    trainer.plot_loss_epoch()         
#print(T.cuda.is_available())
#print(os.environ.get('CUDA_PATH'))
#print(T.__version__)