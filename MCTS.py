from Game import *
from collections import defaultdict
import numpy as np
class MCTS():
    """
    Class for containing the Monte Carlo Tree
    """

    def __init__(self,root):
        self.root_state = root
        self.node_count = 0
        self.root_node = Node(self.root_state)

    def search(self):
        pass

    def expand(self,node):
        pass

    


class Node():
    """
    Class representing the a node in the MCT
    N = number of time this node has been visited
    n = number of times the parent node has been visited
    v = the explotation factor of the node
    """
    def __init__(self,state,parent=None):
        self.state = state
        self.children = set()
        self.parent = parent
        self.N = 0
        self.win = 0
    
    def add_children(self,children):
        for child in children:
            self.children.add(child)
    
    def value(self,exploration_factor):
        """
        Calculates the Upper Confidence Bound for The node
        """
        if self.N == 0:
            return 0 if exploration_factor == 0 else np.inf
        else:
            return self.win/self.N + (exploration_factor*np.sqrt(2*np.log(self.parent.N)/self.N))

