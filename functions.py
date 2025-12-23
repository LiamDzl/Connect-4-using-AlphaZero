import torch
from connect_4 import Grid, graphic
import copy
from rich import print as richprint
from rich.panel import Panel
from rich.columns import Columns
from rich.padding import Padding

def softmax_temp(dist, temp):
    if temp == 0:
        index = torch.argmax(dist, dim=1)
        output = torch.zeros(1, 7)
        output[0, index] = 1
        return output

    else:
        dist = dist.float()
        mask = (dist == 0)
        logits = torch.log(dist + 1e-12) / temp
        logits = logits - torch.max(logits)
        output = torch.softmax(logits, dim=1)
        output[mask] = 0
        return output

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
                               [0,0,0,0,0,0,0]]))
        grid.action(column=i)
        output.append(grid.state)

    for i in range(7):

        leaf = output[i+1] # one of the seven with one red move
        
        for j in range(7):
            leaf = copy.deepcopy(leaf)
            leaf_grid = Grid(state=leaf)
            leaf_grid.action(column=j)
            output.append(leaf_grid.state)
            

    return output


def bar_matrix_emoji(dist, height, colour):
    space = " "
    # Normalize
    max_val = dist.max().item()
    if max_val == 0:
        heights = [0] * len(dist)
    else:
        heights = [int((v / max_val) * height) for v in dist]

    # Build matrix top-down
    rows = []
    for row in range(height, 0, -1):
        line = ""
        for index, h in enumerate(heights):
            if h >= row:
                line += f"{colour} "
            else:
                line += "⬛ "
        rows.append(line.rstrip())

    return "\n".join(rows)

def alphazero_display(policy_name, state, tree_dist, tree_value, neural_dist, neural_value, agent_move, noise):
    print("")
    print("+ ------------ AlphaZero UI ------------- +")
    print("")
    print(f"<state>:")
    print("")
    graphic(state)
    print("")
    print(f"<model>: {policy_name}")
    print("")
    print(f"<\dist\\tree>: {tree_dist}")
    print(f"<\dist\\neural>: {neural_dist}")
    print("")

    if tree_value >= 0:
        richprint(f"<\\value\\tree>: [bold green]{tree_value}")
    else:
        richprint(f"<\\value\\tree>: [bold red]{tree_value}")

    if neural_value.item() >= 0:
        richprint(f"<\\value\\neural>: [bold green]{neural_value}")
    else:
        richprint(f"<\\value\\neural>: [bold red]{neural_value}")
    print("")
    print(f"<\decision\\tree>:      <\decision\\neural>:")
    print("")
    g1 = bar_matrix_emoji(tree_dist[0], height=6, colour="⬜")
    g2 = bar_matrix_emoji(neural_dist, height=6, colour="⬜")
    output = Columns([g1, Padding(g2, (0, 0, 0, 2))])
    richprint(output)
    print("")
    print(f"# 🌿 Tree w/ Temperature:")
    print(softmax_temp(tree_dist, temp=noise))
    print("")
    print(f"# Agent's Choice: {agent_move}")
    print("")



