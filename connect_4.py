import numpy as np
import torch

def mask(state):
    result = [True, True, True, True, True, True, True, False] # 8th corresponds to value, so to be removed anyway
    for i in range(7):
        if state[0, i] != 0:
            result[i] = False   

    result = torch.tensor(result)
    result = result.reshape(1,8)
    return result

class Grid():
    def __init__(self, state): # state a 6 x 7 tensor
        self.state = state

    def action(self, player, column):
        if self.state[0, column] != 0:
            return "### Illegal Action ###"
        
        else:
            for i in range(6):
                check = self.state[5-i, column]
                if int(check.item()) == 0:
                    self.state[5-i, column] = player
                    break

            return "### Legal Action ###"
        

def winner(state): # 6+6 diags, 6 rows and 7 columns to check - id first make them vectors in own right (easy), givem names like TL_BR_4 , and from there check if four in a row
    # Check rows   
    for i in range(6):
        row = state[i]
        for j in range(row.size(0) - 3):
            if row[j] == row[j+1] and row[j+1] == row[j+2] and row[j+2] == row[j+3] and int(row[j].item()) != 0:
                return int(row[j].item())

    # Check columns   
    for i in range(7):
        column = state[:, i]
        for j in range(column.size(0) - 3):
            if column[j] == column[j+1] and column[j+1] == column[j+2] and column[j+2] == column[j+3] and int(column[j].item()) != 0:
                return int(column[j].item())
         
    # Check " \ " diagonals

    for i in range(6):
        diagonal = torch.diagonal(state, offset=i-2, dim1=0, dim2=1)
        for j in range(diagonal.size(0) - 3):
            if diagonal[j] == diagonal[j+1] and diagonal[j+1] == diagonal[j+2] and diagonal[j+2] == diagonal[j+3] and int(diagonal[j].item()) != 0:
                return int(diagonal[j].item())
    
    # Check " / " diagonals

    mirror_state = torch.flip(state, dims=[1]) # Flip matrix to check " / " diagonals

    for i in range(6):
        diagonal = torch.diagonal(mirror_state, offset=i-2, dim1=0, dim2=1)
        for j in range(diagonal.size(0) - 3):
            if diagonal[j] == diagonal[j+1] and diagonal[j+1] == diagonal[j+2] and diagonal[j+2] == diagonal[j+3] and int(diagonal[j].item()) != 0:
                return int(diagonal[j].item())
            
    return 0

def graphic(state):
    for row in state:
        for j in row:
            if int(j.item()) == 1:
                print("🔴", end=" ")
            elif int(j.item()) == -1:
                print("🟡", end=" ")
            elif int(j.item()) == 0:
                print("⚫", end=" ")

        print("")