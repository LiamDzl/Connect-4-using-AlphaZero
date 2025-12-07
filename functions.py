import torch
from connect_4 import Grid, graphic
import copy

def colour(x):
    if x == 1:
        return "Red"
    else:
        return "Yellow"
    
def generate_57():
    output  = []
    output.append(torch.tensor([[0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0]]))

    for i in range(7):
        grid = Grid(state=torch.tensor([[0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0]]), player=1)
        grid.action(column=i)
        output.append(grid.state)

    for i in range(7):

        leaf = output[i+1] # one of the seven with one red move
        
        for j in range(7):
            leaf = copy.deepcopy(leaf)
            leaf_grid = Grid(state=leaf, player=-1)
            leaf_grid.action(column=j)
            output.append(leaf_grid.state)
            

    return output
