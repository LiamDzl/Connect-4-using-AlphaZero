import torch
from connect_4 import Grid, mask, winner, graphic
import math
import numpy as np
import copy

exploration_constant = 1.2

def PUCT(parent, child, player):

    prior_score = child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    # Note, we use parent.visit_count + 1 since the fact we're even computing this for a parent-child pair means parent was visited

    prior_score = player * prior_score # Flip as to incentivise yellow to explore NEGATIVE valuations for red
    prior_score = exploration_constant * prior_score

    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    return value_score + prior_score # Scalar, flipped if yellow perspective

class Node:
    def __init__(self, prior, player, model):
        self.prior = prior # Parent's probability viewpoint of choosing child (scalar)
        self.player = player
        self.model = model

        self.state = None # 6 x 7 tensor
        self.nn_dist = None # 1 x 7 tensor
        self.value_sum = 0
        self.visit_count = 0
        self.parent = None # Stores single parent node
        self.parent_action = None
        self.children = [None, None, None, None, None, None, None] # Stores children node objects
        self.is_terminal = False

    def expand(self, state):
         # Once node is created, all we have is scalar prior and whose turn it is...
         # expanding only necessary if we actually decide to explore this action

         self.state = state
         state = state.reshape(42)
         player = torch.tensor([self.player])
         player = player.reshape(1)

         x = torch.cat((state, player), dim=0)
         x = x.float()

         output_vector = self.model.forward(x)
         output_vector.detach()

         # Grab relevant infos
         nn_dist = output_vector[0:7]
         nn_value = output_vector[7] 

         # Now create set of "ghost nodes" for all non-zero probabilities -- 
         # these represent actions only, we've not computed any of these nodes out properly
         
         for index, probability in enumerate(nn_dist):
            if probability != 0:
                self.children[index] = Node(prior=probability, player=self.player * -1, model=self.model)
                self.children[index].parent = self
                self.children[index].parent_action = index

                # Note child's index in children list represents which action was taken to get from parent to child

         return nn_value, nn_dist

    def expanded(self):
            if self.children == [None, None, None, None, None, None, None]:
                return False
            else:
                return True

    def value(self):
        if self.visit_count == 0:
             return 0
        return self.value_sum / (self.visit_count)
    
    def best_child(self, player):
        current_selection = None
        for child in self.children:
              if child is not None:
                current_selection = child
                break
        
        for child in self.children: # Compare all children PUCT scores
            if child == None:
                pass
            elif player == 1:
                if PUCT(self, child, player=self.player) > PUCT(self, current_selection, player=self.player):
                    current_selection = child
            elif player == -1:
                if PUCT(self, child, player=self.player) < PUCT(self, current_selection, player=self.player):
                    current_selection = child

        return current_selection
    
    def reset_root(self, state):
         self.state = state.detach().clone()

class MCTS:
     def __init__(self, model, iterations):
          self.model = model
          self.iterations = iterations
        
          self.explored_nodes = [] # Maintain a list of all nodes
        
     def run(self, state, player, display):
          total_wins = 0
          red_wins = 0
          yellow_wins = 0

          root = Node(prior=None, player=player, model=self.model)
          root.expand(state=state)
          self.explored_nodes.append(root)

          ### Stats for Testing ##########################################################################################################################
                    
          if display == True:
            
            print(f"### Statistics for Root Node ###\n")
            for index, child in enumerate(root.children):
                    print(f"""Column {index+1}.\n Visits: {f"{child.visit_count:.{5}f}"}, 
                        Average Value:     {f"{child.value():.{5}f}"},
                        Low Visit Boost:   {f"{exploration_constant * child.prior * math.sqrt(root.visit_count + 1) / (child.visit_count + 1):.{5}f}"},
                        Final PUCT:        {f"{PUCT(root, child, player=player):.{5}f}"}""")


          ################################################################################################################################################

          for iteration in range(self.iterations):
                proceed = True
                # Expand Root
                if display == True:
                    print(f"\n ### Iteration {iteration} ###\n")
                
                depth = 0
                root.reset_root(state)
                current_node = root
                path = [current_node]
              
                # Walk down path, storing its for later backprop
                while current_node.expanded() == True:
                        
                    snapshot = current_node.state
                    snap_player = current_node.player
                    current_node = current_node.best_child(player=current_node.player)

                    path.append(current_node)
                    depth += 1

                    if current_node.is_terminal == True: # So that expansion stage is ignored
                        proceed = False
                        break

                    self.explored_nodes.append(current_node)
                    player = player * -1

                    if display == True:
                        graphic(snapshot)
                        print(f"\n ... with player {snap_player} to move\n")


                # Select and Expand
                if proceed == True:
                    parent_node = current_node.parent
                    grid = Grid(state=parent_node.state,
                                player=parent_node.player)
                
                    
                    state_save = copy.deepcopy(parent_node.state) # Debugging Error
                    grid.action(column=current_node.parent_action) # Gives new state for unexpanded node
                    parent_node.state = state_save
                
    
                    nn_value, nn_dist = current_node.expand(state=grid.state) # Expand node, grab value to backprop
        

                    nn_value = nn_value.item() # Remove tensor coat
                    # Check if terminal
                    
                if winner(current_node.state) != 0:
                    current_node.is_terminal = True
                    colour = winner(current_node.state)
                    nn_value = 1 # Fix signal - backprop will handle which nodes are assigned +/- 1
      
                    if colour == 1: # i.e. a win,
                            red_wins += 1
                    else:
                            yellow_wins += 1
                    total_wins += 1

                # Backprop
                for node in path:
                    node.visit_count += 1
                    node.value_sum += -1 * current_node.player * nn_value # either direct NN value, or terminal reward
                    # Notice current_node.player either +/- 1. This means that when node is from yellow's perspective, values are flipped

          ### Stats for Testing ##########################################################################################################################
                    
                    if display == True:
                        print(f"\n🌿🌿🌿 Leaf Node Selected at Depth = {depth} 🌿🌿🌿\n")
                        graphic(current_node.state)
                        print(f"\n ... with value {nn_value} and distribution {nn_dist}\n")
                        print("# # # # # # # # # # # # # # # # # # #")
                        print(f"\n### Statistics for Root Node ###\n")
                        for index, child in enumerate(root.children):
                             print(f"""Column {index+1}.\n Visits: {f"{child.visit_count:.{5}f}"}, 
                                   Average Value:     {f"{child.value():.{5}f}"},
                                   Low Visit Boost:   {f"{exploration_constant * child.prior * math.sqrt(root.visit_count + 1) / (child.visit_count + 1):.{5}f}"},
                                   Final PUCT:        {f"{PUCT(root, child, player=1):.{5}f}"},
                                   Best Child:        {(root.best_child(player=1)).parent_action + 1}""")           

          ################################################################################################################################################
          
          # Return final counts

          mcts_distribution = torch.zeros(1, 7)
          for child_index, child in enumerate(root.children):
               if child != None:
                    mcts_distribution[0, child_index] = child.visit_count / self.iterations
                    # Normalise visit counts to yield new distribution
        
          if display == True:
            print(f"\n### Final Distribution ###\n")
            print(mcts_distribution)
            print(f"\n### Total Wins: {total_wins} / {self.iterations} ###\n")
            print(f"... With Red Winning {red_wins}, vs Yellow Winning {yellow_wins}\n")

          return mcts_distribution
     
