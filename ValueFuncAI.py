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
import sklearn
from sklearn.preprocessing import OneHotEncoder

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

            

class ValueFuncANN(nn.Module):
    def __init__(self):
        super(ValueFuncANN, self).__init__()        
        self.fc1 = nn.Linear(in_features=325, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.value_head = nn.Linear(in_features=64, out_features=1)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.epsilon = 0.999
        self.epsilon_min = 0.01
        self.optimizer = optim.Adam(self.parameters(),lr=0.01)


    def forward(self,x):
        with T.autograd.set_detect_anomaly(True):
            x = x.to(self.device)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            value_logit = self.value_head(x)
        return T.tanh(value_logit) 

class ValueFuncCNN(nn.Module):
    def __init__(self):
        super(ValueFuncCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, (3,3), stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3,3), stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.flat = nn.Flatten()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        x = T.randn(1,2,5,5)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.loss = nn.MSELoss()
        self.to(self.device)
        self.epsilon = 0.999
        self.epsilon_min = 0.01
        self.optimizer = optim.Adam(self.parameters(),lr=0.01)


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
            x = x.reshape(1,2,5,5).float().to(self.device)
            x = self.convs(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x

class Agent_ANN():
    def __init__(self, name, explore,model=None):
        self.nn = model if model != None else ValueFuncANN()
        self.name = name
        self.workers = [Worker([], str(self.name)+"1"), Worker([], str(self.name)+"2")]
        self.explore = explore
        self.loss_array = []
        self.mappings = {
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
    
    def action(self, board):
        states = board.all_possible_next_states(self.name)
        values = []
        rand = np.random.uniform()
        for state in states:
            converted_state = self.convert_nodes_to_input(state)
            values.append(T.flatten(self.nn.forward(converted_state).to(self.nn.device)))
        if (self.explore == False):
            highest_value = T.argmax(T.cat(values)).item()
            return states[highest_value]
        else:
            if (rand > self.nn.epsilon):
                highest_value = T.argmax(T.cat(values)).item()
                return states[highest_value]
            else:
                choice = random.choice(values)
                index = values.index(choice)
                return states[index]

    def convert_nodes_to_input(self, board):
        """
        Converts a set of nodes to a list of one hot encoded boards
        """
        
        enc = OneHotEncoder(handle_unknown='ignore')
        vals = np.array(list(self.mappings.values())).reshape(-1,1)
        enc.fit(vals)
        
        in_nn = []
        
        for row in board.board:
            temp1 = []
            for element in row:
                worker = element.worker.name[0] if element.worker != None else None
                temp1.append([self.mappings[(element.building_level,worker)]])
            one_hot = np.array(enc.transform(np.array(temp1, np.float)).toarray(), np.float)
            in_nn.append(one_hot)
        in_nn = np.array(in_nn).flatten()
        return T.as_tensor(in_nn).float()
        
    
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

class Agent_CNN():
    def __init__(self, name, explore,model=None):
        self.nn = model if model != None else ValueFuncANN()
        self.name = name
        self.workers = [Worker([], str(self.name)+"1"), Worker([], str(self.name)+"2")]
        self.explore = explore
        self.loss_array = []
     
    def action(self, board):
        states = board.all_possible_next_states(self.name)
        values = []
        rand = np.random.uniform()
        for state in states:
            converted_state = self.convertTo2D(state)
            values.append(T.flatten(self.nn.forward(converted_state).to(self.nn.device)))
        if (self.explore == False):
            highest_value = T.argmax(T.cat(values)).item()
            return states[highest_value]
        else:
            if (rand > self.nn.epsilon):
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
        return T.as_tensor(data).double()
        
    
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


    

           