import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from Game import HumanPlayer,Board
from MCTS_NN import Neural_Network
from MCTS import MCTS,Node
from RandomAgent import RandomAgent
from tqdm import tqdm
from linear_rl_agents import LinearRlAgentV2
from ValueFuncAI import ValueFunc

class MCTS_Agent(HumanPlayer):
    def __init__(self,player):
        super().__init__(player)
        self.nn = Neural_Network()
        try:
            self.nn.load_state_dict(r"C:\Users\sarya\Documents\GitHub\Master-Procrastinator\MCTS_AI")
            self.nn.eval()
        except:
            print("Model Dictionary not found")

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
        if board.Player_turn == "A":
            return states[np.argmax(values)]
        else:
            return states[np.argmin(values)]
    
    def convert_nodes_to_input(self,set_of_nodes):
        """
        Converts a set of nodes to a list of one hot encoded boards
        """
        states = [i.state for i in set_of_nodes]
        boards = [i.board for i in states]
        
        enc = OneHotEncoder(handle_unknown='ignore')
        vals = np.array(list(self.mappings.values())).reshape(-1,1)
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



class Trainer():
    def __init__(self,args,NN = None):
        self.args = args
        self.state = Board(RandomAgent("A"),RandomAgent("B"))
        self.training_examples = []
        self.mcts = None
        self.nn = NN if NN != None else Neural_Network()
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
        self.nn.to(self.nn.device)
    
    def initialize_mcts(self):
        self.state.PlayerA.place_workers(self.state)
        self.state.PlayerB.place_workers(self.state)
        root = Node(self.state)
        self.mcts = MCTS(root,self.nn,self.args)

    def convert_nodes_to_input(self,set_of_nodes):
        """
        Converts a set of nodes to a list of one hot encoded boards
        """
        states = [i.state for i in set_of_nodes]
        boards = [i.board for i in states]
        
        enc = OneHotEncoder(handle_unknown='ignore')
        vals = np.array(list(self.mappings.values())).reshape(-1,1)
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
        print("\nGenerating Data")
        training_data = []
        temp_MCTS = self.mcts
        node = self.mcts.root
        for i in tqdm(range(self.args['depth'])):
            root = temp_MCTS.breadth_run(node)
            app = list(temp_MCTS.collapse(root))
            training_data+=app
            node = root.select_child()

        return training_data
    
    def learn(self,train_examples):
        """
        Learn using One MCTS tree
        """
        print("\nLearning from Data")
        boards = self.convert_nodes_to_input(train_examples)
        target_values = [node.value() for node in train_examples]
        data = [(boards[i],target_values[i]) for i in range(len(boards))]
        np.random.shuffle(data)
        
        for i in range(len(boards)):
            target = torch.tensor(data[i][1],dtype=torch.float32).to(self.nn.device)
            target = target.view(1)
            temp = torch.from_numpy(data[i][0]).float().to(self.nn.device)
            pred = self.nn.forward(temp).to(self.nn.device)
            loss = self.nn.loss(pred,target)
            self.nn.optimizer.zero_grad()
            loss.backward()
            self.nn.optimizer.step()
            self.loss_array.append(loss.item())

        self.plot_loss()

    def train(self):
        self.loss_array = []
        for i in tqdm(range(self.args["epochs"])):
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
        torch.save(self.nn.state_dict(),filepath)
   
    
    def plot_loss(self):
        plt.plot(self.loss_array)
        plt.title("Loss versus iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()




class Trainer_CNN(Trainer):

    def __init__(self,args,NN=None):
        self.args = args
        self.state = Board(LinearRlAgent("A"),LinearRlAgent("B"))
        self.training_examples = []
        self.mcts = None
        self.nn = NN if NN != None else ValueFunc()
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
        self.nn.to(self.nn.device)

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
                elif square.worker.name[0] == "A":
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(1)
                else:
                    temp_lst.append(square.building_level/4)
                    temp_lst2.append(-1)
            buildings.append(temp_lst)
            players.append(temp_lst2)
        data.append(buildings)
        data.append(players)
        return torch.as_tensor(data)

    def convert_nodes_to_training_data(self, set_of_nodes):
        training_data = [(i.state,i.value()) for i in set_of_nodes]
        shuffle(training_data)
        return training_data
    
    def generate_training_data(self):
        """
        Perform iteration of MCTS and return a collapsed tree for training
        """
        print("\nGenerating Data")
        training_data = []
        temp_MCTS = self.mcts
        node = self.mcts.root
        for i in tqdm(range(self.args['depth'])):
            root = temp_MCTS.breadth_run(node)
            app = list(temp_MCTS.collapse(root))
            training_data+=app
            node = root.select_child()

        return training_data

    def save_checkpoint(self,folder):
        """
        Save the Neural Network
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        filepath = os.path.join(folder,"MCTS_AI_CNN")
        torch.save(self.nn.state_dict(),filepath)
    
    def learn(self,train_examples):
        """
        Learn using One MCTS tree
        """
        print("\nLearning from Data")
        
        for i in range(len(train_examples)):
            target = torch.tensor(train_examples[i][1],dtype=torch.float32).to(self.nn.device)
            target = target.view(1)
            converted_state = self.convertTo2D(train_examples[i][0])
            pred = t.nn.forward(converted_state).to(t.nn.device)
            loss = self.nn.loss(pred,target)
            self.nn.optimizer.zero_grad()
            loss.backward()
            self.nn.optimizer.step()
            self.loss_array.append(loss.item())

        self.plot_loss()

    def train(self):
        self.loss_array = []
        for i in tqdm(range(self.args["epochs"])):
            training_examples = self.generate_training_data()
            training_examples = self.convert_nodes_to_training_data(training_examples)
            self.learn(training_examples)
        self.save_checkpoint(r'C:\Users\sarya\Desktop\Semester 4\ISM\Game')
        pass
 
    def action(self, board):
        states = board.all_possible_next_states(self.name)
        values = []
        rand = 1 #np.random.uniform()
        for state in states:
            converted_state = self.convertTo2D(state)
            values.append(T.flatten(self.nn.forward(converted_state).to(self.nn.device)))
        highest_value = torch.argmax(T.cat(values)).item()
        return states[highest_value]