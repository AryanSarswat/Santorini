from elo import rate_1vs1   
from random import choice,shuffle
import matplotlib.pyplot as plt
import time
"""
all_games format

[(A,B),(C,D)]

where (A,B) is a game played where A won

info = 
{ A,B : 100
    B,C : 20
    C,D : 5
}

 games = []

 for i,j in info.items():
     for k in range(j):
         games.append(i)
"""

args = {
    'Num_Simulations': 1,
    'Iterations' : 10000,
    'Tree_depth' : 2,                     # Total number of MCTS simulations to run when deciding on a move to play
    'epochs': 1,
    'depth' : 25,                                    # Number of epochs of training per iteration
    'checkpoint_path': r"C:\Users\sarya\Documents\GitHub\Master-Procrastinator",
    'random' : 0
}

class Bot():
    def __init__(self,name):
        self.name = name
        self.elo = 1000
        self.past = []
    
    def update_elo(self,new_elo):
        self.past.append(self.elo)
        self.elo = new_elo

Linear_Manual= Bot("Linear_Manual")
Random = Bot("Random")
MCTS_ANN = Bot("MCTS_ANN")
MCTS_CNN = Bot("MCTS_CNN")
MCTS = Bot("MCTS")
ANN = Bot("ANN")
CNN = Bot("CNN")
Combined = Bot("Combined")
Linear_Rootstrap = Bot("Linear_Rootstrap")
Linear_Treestrap = Bot("Linear_Treestrap")

games = {
    (Linear,Random) : 100,
    (MCTS_ANN,Random) : 99,
    (Random,MCTS_ANN) : 1,
    (MCTS_CNN,Random) : 100,
    (MCTS,Random) : 100,
    (Linear,MCTS_ANN) : 98,
    (MCTS_ANN,Linear) : 2,
    (Linear,MCTS_CNN) : 89,
    (MCTS_CNN,Linear) : 11,
    (MCTS,MCTS_ANN) : 98,
    (MCTS_ANN,MCTS) : 2,
    (MCTS,MCTS_CNN) : 89,
    (MCTS_CNN,MCTS) : 11,
    (MCTS_CNN,MCTS_ANN) : 84,
    (MCTS_ANN,MCTS_CNN) : 16,
    (Linear,MCTS) : 79,
    (MCTS,Linear) : 21,
    (CNN,Random) : 98,
    (Random,CNN) : 2,
    (Linear,CNN) : 84,
    (CNN,Linear) : 16
}

all_games = []
for i,j in games.items():
    for k in range(j):
        all_games.append(i)


shuffle(all_games)

def sim(all_games):
    while len(all_games) != 0:
        #Remove one random game
        game = all_games.pop(all_games.index(choice(all_games)))
        
        #Rating Update for winner
        A,B = game[0],game[1]
        rat_a,rat_b = rate_1vs1(A.elo,B.elo)
        if rat_b < 0:
            rat_b = 0
        A.update_elo(rat_a)
        B.update_elo(rat_b)
    

def sim_time(all_games):
    time_end = time.time() + 30
    while time.time() < time_end:
        #Remove one random game
        game = choice(all_games)
        
        #Rating Update for winner
        A,B = game[0],game[1]
        rat_a,rat_b = rate_1vs1(A.elo,B.elo)
        if rat_b < 0:
            rat_b = 0
        A.update_elo(rat_a)
        B.update_elo(rat_b)


if __name__=='__main__':
    sim_time(all_games)

    fig,ax = plt.subplots(figsize=(14,12))
    ax.set_xlabel("Time")
    ax.set_ylabel("ELO")
    ax.plot(Linear.past,label="Linear")
    ax.plot(Random.past,label="Random")
    ax.plot(MCTS_ANN.past,label="MCTS_ANN")
    ax.plot(MCTS_CNN.past,label="MCTS_CNN")
    ax.plot(MCTS.past,label="MCTS")
    ax.plot(CNN.past,label = "CNN")
    ax.legend()
    plt.show()
    print(f"\nLinear Final ELO is : {Linear.elo}\nRandom Final ELO is : {Random.elo}\nMCTS_ANN Final ELO is : {MCTS_ANN.elo}\nMCTS_CNN Final ELO is : {MCTS_CNN.elo}\nMCTS Final ELO is : {MCTS.elo}\nCNN Final ELO is : {MCTS.elo}\n")
