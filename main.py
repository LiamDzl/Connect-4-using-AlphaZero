import numpy as np
from neural_network import policy
import torch
from connect_4 import Grid, winner, graphic
import time

policy_network = policy(structure=[42,100,100,42,8])

x = torch.zeros(6,7)
environment = Grid(state=x)

column = ""
print("\n### Connect 4 ###\n")
print("(Enter \"end\" to exit)\n")

graphic(environment.state)
print("")

move = 0

while (column != "end"):

    proceed = False
    inrange = False
    
    while not proceed:
        column = input("🔴 Select any column from 1 to 7: ")
        print(f"\n----------------------- ( Move {move} )")
        colNum = -1
        if column == "end":
            
            break
        try:
            colNum = int(column)
        except:
            print("\n❌ ERROR: not in range\n")

        if colNum >= 0:
            if colNum >= 1 and colNum <= 7:
                proceed = True
                inrange = True

            if inrange == True:
                if environment.state[0, int(column) - 1] != 0:
                    print("\n❌ ERROR: column full!\n")
                    proceed = False
            
            else:
                print("\n❌ ERROR: not in range\n") 
            

    if column == "end":
        break

    else:
        environment.action(player=1, column=int(column)-1)
        move += 1

        if winner(environment.state) == 1:
            print("\n🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
            graphic(environment.state)
            column = "end"

        print("\n")

        if column == "end":
            break

        else:
            graphic(environment.state)
        
        value, distribution = policy_network.forward(environment.state)
        distribution = distribution.detach()
        time.sleep(0.8)

        yellow_sample = torch.multinomial(distribution, num_samples=1)

        print("\n")
        environment.action(player=-1, column=int(yellow_sample.item()))
        print("Value Function: ", value.item())
        print("Distribution: ", distribution)
        print("\n")
        graphic(environment.state)
        print("\n")

        if winner(environment.state) == -1:
            print("🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
            column = "end"

