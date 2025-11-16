import numpy as np
from neural_network import policy
import torch
from connect_4 import Grid, winner, graphic
import time

# policy_network = model(structure=[42,100,100,42,8,7], # 7 scalars for distibution, and the last scalar in the penultimate vector encodes value function
#            activation="softmax")

x = torch.zeros(6,7)
environment = Grid(state=x)

column = ""
print("\n### Connect 4 ###\n")
print("(Enter \"end\" to exit)\n")

graphic(environment.state)
print("")

while (column != "end"):
    column = input("🔴 Select any column from 1 to 7: ")

    if column == "end":
        break

    else:
        print("\n")
        environment.action(player=1, column=int(column)-1)

        if winner(environment.state) == 1:
            print("\n🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
            print("\n")
            column = "end"
        
        graphic(environment.state)
        
        if column == "end":
            break

        time.sleep(0.8)
        yellow = np.random.randint(7)
        print("\n")
        environment.action(player=-1, column=int(yellow))
        graphic(environment.state)
        print("\n")

        if winner(environment.state) == -1:
            print("\n🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
            column = "end"

