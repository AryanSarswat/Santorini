import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Game import *
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Game import *
from RandomAgent import *
import os

def state_mappings():
    mappings = {
        (0,None) : 0,
        (1,None) : 1,
        (2,None) : 2,
        (3,None) : 3,
        (4,None) : 4,
        (0,'A') : 5,
        (1,'A') : 6,
        (2,'A') : 7,
        (3,'A') : 8,
        (0,'B') : 9,
        (1,'B') : 10,
        (2,'B') : 11,
        (3,'B') : 12,
    }
    return mappings

class Logger():
    def __init__(self):
        self.values = []
            

class ValueFunc(nn.Module):
    def __init__(self):
        super(ValueFunc, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, (3,3), stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3,3), stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.flat = nn.Flatten()

        x = T.randn(1,2,5,5)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 32)
        self.fc2 = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(),lr=0.01)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0')
        self.to(self.device)
        self.epsilon = 0.999
        self.epsilon_min = 0.01
        

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = self.flat(x)

        if self._to_linear == None:
           self._to_linear = x.shape[1]
        return x

    def forward(self,x):
        with T.autograd.set_detect_anomaly(True):
            x = x.reshape(1,2,5,5).to(self.device)
            #x = T.cuda.FloatTensor(x).to(self.device)
            x = self.convs(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x

class Agent():
    def __init__(self, name, explore):
        self.nn = ValueFunc()
        self.name = name
        self.workers = [Worker([], str(self.name)+"1"), Worker([], str(self.name)+"2")]
        self.explore = explore
        self.loss_array = []
        self.all_values = []
    
    def action(self, board):
        states = board.all_possible_next_states(self.name)
        values = []
        rand = 1 #np.random.uniform()
        for state in states:
            converted_state = self.convertTo2D(state)
            values.append(T.flatten(self.nn.forward(converted_state).to(self.nn.device)))
        if (self.explore == False) and (rand > self.nn.epsilon):
            highest_value = T.argmax(T.cat(values)).item()
            return states[highest_value]
        else:
            choice = random.choice(values)
            index = values.index(choice)
            return states[index]


    def convertTo2D(self, board):
        """
        Takes in a board and converts it into 2D tensor form with shape (2, 5, 5)
        """
        data = []
        buildings = []
        players = []
        for squares in board.board:
            temp_lst = []
            temp_lst2 = []
            for square in squares:
                if square.worker == None:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(0)
                elif square.worker.name[0] == self.name:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(1)
                else:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(-1)
            buildings.append(temp_lst)
            players.append(temp_lst2)
        data.append(buildings)
        data.append(players)
        return T.as_tensor(data)
        
    
    def place_workers(self, board):
        """
        Method to randomly place agent's workers on the board
        """
        place_count = 0
        while place_count < 2:
            try:
                coords = [np.random.randint(0, 5), np.random.randint(0, 5)]
                # Updates worker and square
                self.workers[place_count].update_location(coords)
                board.board[coords[0]][coords[1]].update_worker(self.workers[place_count])
                place_count += 1
            except Exception:
                continue
        return board

    def reward(self, win):
        if win == self.name:
            r = 1
        else:
            r = -1
        return r

    def plot_loss(self):
        plt.plot(self.loss_array)
        plt.title("Loss versus iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()


    

           