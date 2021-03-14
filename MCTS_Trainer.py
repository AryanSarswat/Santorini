import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder

class Agent():
    def __init__(self, game, model, args):
        self.game = game
        self.nn = Neural_Network()
        self.args = args
        self.mcts = MCTS(Node(self.game), self.nn, self.args)
        self.name = "A"
        self.workers = [Worker([], str(self.name)+"1"), Worker([], str(self.name)+"2")]
        self.loss_array = []
        self.mappings = mappings = {
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
    
    def action(self, board):
        """
        Performs an action
        """
        states = board.all_possible_next_states(self.name)
        values = []
        for state in states:
            converted_state = self.convert_nodes_to_input(state)
            values.append(self.nn.forward(converted_state))
        return states[np.argmax(values)]
           
    def convert_nodes_to_input(self,set_of_nodes):
        """
        Converts a set of nodes to a list of one hot encoded boards
        """
        states = [i.state for i in set_of_nodes]
        boards = [i.board for i in states]
        
        enc = OneHotEncoder(handle_unknown='ignore')
        vals = np.array(list(mappings.values())).reshape(-1,1)
        enc.fit(vals)
        
        in_nn = []
        for board in boards:
            temp1 = []
            for row in board:
                temp2 = []
                for element in row:
                    worker = element.worker.name[0] if element.worker != None else None
                    temp2.append([self.mappings[(element.building_level,worker)]])
                one_hot = enc.transform(np.array(temp2)).toarray()
                temp1.append(one_hot)
            flattened = np.array(temp1).flatten()
            in_nn.append(flattened)
        return in_nn
    
    def generate_training_data(self):
        """
        Perform iteration of MCTS and return a collapsed tree for training
        """
        print("Generating Data")
        for i in range(self.args['numIters']):
            self.mcts.run(self.mcts.root.state.Player_turn())
        training_data = self.mcts.collapse()
        return training_data
    
    def learn(self,train_examples):
        """
        Learn using One MCTS tree
        """
        print("Learning from Data")
        boards = self.convert_nodes_to_input(train_examples)
        target_values = [node.value() for node in train_examples]
        data = [(boards[i],target_values[i]) for i in range(len(boards))]
        data = np.random.shuffle(data)
        
        for i in range(len(boards)):
            target = torch.tensor(data[i][1],dtype=torch.float32).to(self.nn.device)
            target = target.view(1)
            pred = self.nn.forward(data[i][0]).to(self.nn.device)
            loss = self.nn.loss(pred,target)
            self.nn.optimizer.zero_grad()
            loss.backward()
            self.nn.optimizer.step()
            self.loss_array.append(loss.item())

        self.plot_loss()

    def train(self):
        self.loss_array = []
        for i in range(self.args["epochs"]):
            training_examples = self.generate_training_data()
            self.learn(training_examples)
        self.save_checkpoint(r'C:\Users\sarya\Documents\GitHub\Master-Procrastinator')
        pass

    def save_checkpoint(self,folder):
        """
        Save the Neural Network
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        filepath = os.path.join(folder,"MCTS_AI")
        torch.save({
            'state_dict' : self.nn.state_dict(),
        },filepath)
    
    def plot_loss(self):
        plt.plot(self.loss_array)
        plt.title("Loss versus iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()