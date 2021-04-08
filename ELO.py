from elo import rate_1vs1   
from random import choice,shuffle
import matplotlib.pyplot as plt
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

Linear = Bot("Linear")
Random = Bot("Random")
ANN = Bot("ANN")
CNN = Bot("CNN")
MCTS = Bot("MCTS")

games = {
    (Linear,Random) : 100,
    (ANN,Random) : 99,
    (Random,ANN) : 1,
    (CNN,Random) : 100,
    (MCTS,Random) : 100,
    (Linear,ANN) : 98,
    (ANN,Linear) : 2,
    (Linear,CNN) : 89,
    (CNN,Linear) : 11,
    (MCTS,ANN) : 98,
    (ANN,MCTS) : 2,
    (MCTS,CNN) : 89,
    (CNN,MCTS) : 11,
    (CNN,ANN) : 84,
    (ANN,CNN) : 16,
    (Linear,MCTS) : 79,
    (MCTS,Linear) : 21
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
        A.update_elo(rat_a)
        B.update_elo(rat_b)
    

sim(all_games)


fig,ax = plt.subplots()

ax.plot(Linear.past,label="Linear")
ax.plot(Random.past,label="Random")
ax.plot(ANN.past,label="ANN")
ax.plot(CNN.past,label="CNN")
ax.plot(MCTS.past,label="MCTS")
ax.legend()
plt.show()

