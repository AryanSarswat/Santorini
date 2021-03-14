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
            val = 1 if node.to_play == to_play else -1
            node.value_sum += val*value
            node.visit_count+=1
    
    def rollout(self,node):
        """
        Perform a rollout
        """
        state = node.state
        if state.is_terminal():
            return state.reward()
        else:
            while not state.is_terminal():
                actions = state.all_possible_next_states(state.Player_turn())
                action = np.random.choice(actions)                                 
                state = action
                if state.is_terminal():
                    return state.reward()
    
    def run(self,to_play):
        """
        Perform One iteration of Selection,Expand,Rollout
        """
        if not self.root.is_expanded():
            self.root.expand()
        else:
            search_path = [self.root]
            current_node = self.root
            while current_node.is_expanded():
                current_node = self.root.select_child()
                search_path.append(current_node)
            for sim in range(self.args["Num_Simulations"]):
                reward = self.rollout(current_node)
                self.backpropagate(search_path,reward,current_node.state.Player_turn())
            if not current_node.state.is_terminal():
                current_node.expand()
        return self.root

    def breadth_run(self):
        """
        Performs a rollout for all child nodes regardless of UCB score
        """
        if not self.root.is_expanded():
            self.root.expand()
        child_nodes = list(self.root.children)
        for child in child_nodes:
            search_path = [self.root,child]
            for sim in range(self.args["Num_Simulations"]):
                reward = self.rollout(child)
                self.backpropagate(search_path,reward,child.state.Player_turn())
        return self.root
                

    def collapse(self):
        """
        Return a list of all the nodes in the MCTS
        """
        all_nodes = set()
        to_explore = list(self.root.children.keys())
        all_nodes = all_nodes | set(self.root.children.keys())
        while to_explore != []:
            current = to_explore.pop()
            if current.children != {}:
                childs = list(current.children.keys())
                to_explore += childs
                all_nodes = all_nodes | set(childs)
            else:
                continue

        return all_nodes