from Game import *
from collections import defaultdict
import numpy as np


def upper_confidence_bound(node):
    value = node.value()
    explore = 2*np.sqrt(np.log(node.parent.value())/node.visit_count)
    return value+explore

class Node():
    """
    Class representing the a node in the MCT
    """
    def __init__(self,state,parent=None):
        self.state = state
        self.children = {}
        self.parent = parent
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = state.Player_turn()
    
    def add_children(self,children):
        for child in children:
            self.children.add(child)
    
    def value(self):
        """
        Calculates the value
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum/self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

    def select_action(self,temperature):
        """
        Select an action based on visit count and temperature
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())
        if temperature == 0:
            new_state = actions[np.argmax(visit_counts)]
        elif temperature == np.inf:
            new_state = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            new_state = np.random.choice(actions, p=visit_count_distribution)
        
        return new_state
    
    def select_child(self):
        """
        Selects the child with the highest UCB score
        """
        children_nodes = list(self.children.values())
        children = list(self.children.keys())
        UCB_score = list(map(upper_confidence_bound,children))
        return children[np.argmax(UCB_score)]
    
    def expand(self):
        """
        Expand Node
        """
        children = self.state.all_possible_next_states()
        for child in children:
            self.children[child] = Node(child,parent=self)
        pass

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