# Model will be the current neural network (since self play)
# Function: state, action -> deterministically sets new state -> outputs distribution WRT neural network
# Agent is from yellow's perspective (second player)
# IDEA --- have every node store a 7x7 matrix that describes all possible children from that state (and plug in some IDs for discovered states)
# row i will describe yellow having made move i, and column j describes red having made move j

from connect_4 import Grid, mask

class Explored():
    def __init__(self, database): # database will be a LIST, where index corresponds to ID (but shift )
        self.database = database # of the form of NODES, not dicts

class Node():
    def __init__(self, id, state, value, q_values, visits, children_ids, database): # q_values will store vector for all 7 state-action pairs, children list of node IDs in 7x7 matrix
        self.id = id
        self.state = state
        self.value = value
        self.q_values = q_values
        self.visits = visits
        self.children_ids = children_ids # will be all -1, since i want 0 to represent 0 (base) node
        self.database = database # will be instantiated by 'explored' class. THIS is the reference to where node lives

    def backpropagate(self, model, action): # throw neural net policy as the model (amended for deterministic move)
        grid = Grid(self.state) # create Grid class that can be acted on
        mask = mask(self.state)
        updated = grid.action(player=-1, column=action)
        value, distribution = model.forward(state=updated, mask=mask)
        new_q = 0
        count = 0
        for id in self.children_ids[action]:
            if id != -1: 
                new_q += (self.database[id].value)*distribution[count]
    
            count += 1
        
        self.q_values[action] = new_q







