from elo import rate_1vs1   
from random import choice
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

class Bot():
    def __init__(self,name,games):
        self.name = name
        self.elo = 1000
        self.past = []
    
    def update_elo(self,new_elo):
        self.past.append(self.elo)
        self.elo = new_elo
    


def sim(all_games):
    while len(all_games) != 0:
        #Remove one random game
        game = all_games.pop(all_games.index(choice(all_games)))
        
        #Rating Update for winner
        A,B = game[0],game[1]
        rat_a,rat_b = rate_1vs1(A.elo,B.elo)
        A.update_elo(rat_a)
        B.update_elo(rat_b)
    


