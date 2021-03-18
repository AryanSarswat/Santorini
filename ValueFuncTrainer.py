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

PATH = "Value_Func_State_Dict_CNN_A.pt"



class ValueFuncTrainer():
    def __init__(self, epochs, batches, agent):
        self.epochs = epochs
        self.batches = batches
        self.agent = agent
        self.nn = ValueFunc()
        self.state_values = []
    
    def train(self):
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0
            for batch in tqdm(range(self.batches)):
                batch_loss = 0
                winner = self.training_loop(self.agent, RandomAgent("B"))
                #print(f"\n{winner}") 
                #print(f"\n{self.state_values}")
                with T.autograd.set_detect_anomaly(True):
                    while self.state_values != []:
                        self.nn.optimizer.zero_grad()
                        reward = self.agent.reward(winner)
                        loss = self.nn.loss(self.state_values.pop(-1), T.cuda.FloatTensor([[reward]])).to(self.nn.device)
                        loss.backward()
                        batch_loss += loss.item()
                        epoch_loss += batch_loss
                        reward = reward * 0.97
                    self.nn.optimizer.step()
                    self.nn.epsilon = self.nn.epsilon * 0.999 if self.nn.epsilon > self.nn.epsilon_min else self.nn.epsilon_min
                self.agent.loss_array.append(batch_loss)
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
            else:
                if currentPlayer.name == self.agent.name:
                    states = board.all_possible_next_states(self.agent.name)
                    values = []
                    rand = np.random.uniform()
                    for state in states:
                        converted_state = self.agent.convertTo2D(state)
                        values.append(self.nn.forward(converted_state).to(self.nn.device))
                    if (self.agent.explore == False) and (rand > self.nn.epsilon):
                        highest_value = T.argmax(T.cat(values)).item()
                        self.state_values.append(values[highest_value])
                        board = states[highest_value]
                    else:
                        choice = random.choice(values)
                        index = values.index(choice)
                        self.state_values.append(choice)
                        board = states[index] 
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
def evaluate_model(PATH, brain):
    brain.nn.eval()
    count = 0
    total = 0
    with T.no_grad():
        for i in range(100):
            winner = run_santorini(brain, RandomAgent("B"))
            if winner == "A":
                count += 1
                total += 1
            else:
                total += 1
    return count/total


if os.path.isfile(PATH):
    print("\n Loading Saved Model")
    brain = Agent("A", False)
    brain.nn.load_state_dict(T.load(PATH))
    #print(evaluate_model)
    trainer = ValueFuncTrainer(40, 10, brain)

else:
    print("\n Training..........")
    brain = Agent("A", False)
    trainer = ValueFuncTrainer(40, 100, brain)
    trainer.train()
    T.save(trainer.nn.state_dict(),PATH)
    #print(trainer.agent.nn.epsilon)
    brain.plot_loss()
             
#print(T.cuda.is_available())
#print(os.environ.get('CUDA_PATH'))
#print(T.__version__)