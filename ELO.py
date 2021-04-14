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
ANN = Bot("ANN")
CNN = Bot("CNN")
Combined = Bot("Combined")
Linear_Rootstrap = Bot("Linear_Rootstrap")
Linear_Treestrap = Bot("Linear_Treestrap")

games = {
    (ANN,Random) : 92,
    (Random,ANN) : 8,
    (CNN,Random) : 98,
    (Random,CNN) : 2,
    (MCTS_ANN,Random) : 99,
    (Random,MCTS_ANN) : 1,
    (MCTS_CNN,Random) : 100,
    (Combined,Random) : 100,
    (Linear_Treestrap,Random) : 100,
    (Linear_Rootstrap,Random) : 100,
    (Linear_Manual,Random) : 100,
    
    (CNN,ANN) : 60,
    (ANN,CNN) : 40,
    
    (MCTS_ANN,ANN) : 54,
    (ANN,MCTS_ANN) : 46,
    (MCTS_ANN,CNN) : 52,
    (CNN,MCTS_ANN) : 48,
    
    (MCTS_CNN,ANN) : 72,
    (ANN,MCTS_CNN) : 28,
    (MCTS_CNN,CNN) : 67,
    (CNN,MCTS_CNN) : 37,
    (MCTS_CNN,MCTS_ANN) : 84,
    (MCTS_ANN,MCTS_CNN) : 16,
    
    (Linear_Treestrap,ANN): 88,
    (ANN,Linear_Treestrap) : 12,
    (Linear_Treestrap,CNN): 89,
    (CNN,Linear_Treestrap) : 11,
    (Linear_Treestrap,MCTS_ANN): 89,
    (MCTS_ANN,Linear_Treestrap) : 11,
    (Linear_Treestrap,MCTS_CNN): 89,
    (MCTS_CNN,Linear_Treestrap) : 11,
    
    (Linear_Rootstrap,ANN): 92,
    (ANN,Linear_Rootstrap) : 8,
    (Linear_Rootstrap,CNN): 100,
    (Linear_Rootstrap,MCTS_ANN): 91,
    (MCTS_ANN,Linear_Rootstrap) : 9,
    (Linear_Rootstrap,MCTS_CNN): 90,
    (MCTS_CNN,Linear_Rootstrap) : 10,
    (Linear_Rootstrap,Linear_Treestrap) : 53,
    (Linear_Treestrap,Linear_Rootstrap) : 47,

    (Linear_Manual,ANN): 86,
    (ANN,Linear_Manual) : 14,
    (Linear_Manual,CNN): 84,
    (CNN,Linear_Manual) : 16,
    (Linear_Manual,MCTS_ANN): 88,
    (MCTS_ANN,Linear_Manual) : 12,
    (Linear_Manual,MCTS_CNN): 87,
    (MCTS_CNN,Linear_Manual) : 13,
    (Linear_Manual,Linear_Treestrap) : 49,
    (Linear_Treestrap,Linear_Manual) : 51,
    (Linear_Manual,Linear_Rootstrap): 46,
    (Linear_Rootstrap,Linear_Manual) : 14,

    (Combined,ANN) : 94,
    (ANN,Combined) : 6,
    (Combined,CNN): 90,
    (CNN,Combined): 10,
    (Combined,MCTS_ANN) : 89,
    (MCTS_ANN,Combined) : 11,
    (Combined,MCTS_CNN): 85,
    (MCTS_CNN,Combined): 15,

    (Combined,Linear_Treestrap) : 53,
    (Linear_Treestrap,Combined) : 47,

    (Combined,Linear_Rootstrap) : 54,
    (Linear_Rootstrap,Combined) : 46,

    (Combined,Linear_Manual) : 42,
    (Linear_Manual,Combined) : 58,



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
    time_end = time.time() + 60
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
    sim(all_games)

    fig,ax = plt.subplots(figsize=(14,12))
    ax.set_xlabel("Time")
    ax.set_ylabel("ELO")
    ax.plot(Linear_Manual.past,label=Linear_Manual.name)
    ax.plot(Linear_Rootstrap.past,label=Linear_Rootstrap.name)
    ax.plot(Linear_Treestrap.past,label=Linear_Treestrap.name)
    ax.plot(Random.past,label="Random")
    ax.plot(MCTS_ANN.past,label="MCTS_ANN")
    ax.plot(MCTS_CNN.past,label="MCTS_CNN")
    ax.plot(CNN.past,label = "CNN")
    ax.plot(ANN.past,label=ANN.name)
    ax.plot(Combined.past,label=Combined.name)
    ax.legend()
    plt.show()
    print(f"""
    \n{Linear_Manual.name} Final ELO is : {Linear_Manual.elo}
    \n{Linear_Rootstrap.name} Final ELO is : {Linear_Rootstrap.elo}
    \n{Linear_Treestrap.name} Final ELO is : {Linear_Treestrap.elo}
    \n{ANN.name} Final ELO is : {ANN.elo}
    \n{CNN.name} Final ELO is : {CNN.elo}
    \n{MCTS_ANN.name} Final ELO is : {MCTS_ANN.elo}
    \n{MCTS_CNN.name} Final ELO is : {MCTS_CNN.elo}
    \n{Combined.name} Final ELO is : {Combined.elo}
    \n{Random.name} Final ELO is : {Random.elo}
    """)
