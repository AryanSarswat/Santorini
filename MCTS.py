from Game import *
from collections import defaultdict
import numpy as np


def upper_confidence_bound(node):
    """
    Function which return the Upper Confidence Bound
    C = 2 has been implemented to balance exploratio and exploitation
    """
    value = node.value()
    explore = 2*np.sqrt(np.log(node.parent.visit_count)/node.visit_count)
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
        visit_counts = np.array([child.visit_count for child in self.children.keys()])
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
        children_nodes = list(self.children.keys())
        UCB_score = list(map(upper_confidence_bound,children_nodes))
        return children_nodes[np.argmax(UCB_score)]
    
    def expand(self):
        """ 
        Expand Node
        """
        children = self.state.all_possible_next_states(self.to_play)
        for child in children:
            self.children[Node(child,parent=self)] = child
        pass

class MCTS():
    """
    Class for containing the Monte Carlo Tree
    """

    def __init__(self,root,model,args):
        self.root = root
        self.model = model
        self.args = args

    def backpropagate(self,search_path,value,to_play):
        """
        Backpropagate the value of state
        """
        for node in reversed(search_path):
            node.value_sum += 1 if node.to_play == to_play else -1
            node.visit_count+=1
    
    def rollout(self,node):
        state = node.state
        if state.is_terminal():
            return state.reward()
        else:
            while not state.is_terminal():
                action = np.random.choice(state.all_possible_action(state.Player_turn()))
                state = action
                if state.is_terminal():
                    return state.reward()
    
    def run(self,model,state,to_play):
        root = Node(state)
        root.expand()
        for _ in range(self.args["Num_Simulations"]):
            node = root
            search_path = [node]
            #Select
            while node.is_expanded():
                node = node.select_child()
                search_path.append(node)
            
            parent = search_path[-2]
            state = parent.state
            next_state = node
            value = next_state.state.reward()

            if value == 0:
                #Game has not ended thus expand the node
                node.expand()
            
            self.backpropagate(search_path,value,next_state.state.Player_turn())
        
        return root