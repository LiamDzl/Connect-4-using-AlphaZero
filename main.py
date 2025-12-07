import torch
from neural_network import policy
from connect_4 import Grid, winner, graphic
from tree_module import MCTS

torch.serialization.add_safe_globals([policy])
policy_network = torch.load("weird_parameters.pt", weights_only=False)
policy_name = "weird_parameters.pt"
tree_search = MCTS(model=policy_network, iterations=600)

x = torch.zeros(6,7)
environment = Grid(state=x, player=1)

column = ""
move = 0

print("\n### Connect 4 ###\n")
print("(Enter \"end\" to exit)\n")

proceed = False
while not proceed:
    player_colour = input("🔴 or 🟡 ? : ")
    print("")

    if player_colour == "end":
            column = "end"
            break
    
    if player_colour == "red":
        emoji = "🔴"
        player_colour = 1
        agent_colour = -1
        graphic(environment.state)
        print("\n")
        proceed = True

    elif player_colour == "yellow":
        emoji = "🟡"
        player_colour = -1
        agent_colour = 1
        graphic(environment.state)
   
        distribution = tree_search.run(state=environment.state, player=1, display=False)
        print(f"\n### {policy_name}'s TREE Decision Tensor: {distribution}\n")

        state = torch.cat(((environment.state).reshape(42), torch.tensor([1])), dim=0)

        output = policy_network(state)
        neural_distribution = output[:7]
        value = output[7]

        print(f"### {policy_name}'s NEURAL Decision Tensor: {neural_distribution}\n")

        agent_move = torch.argmax(distribution)
        print(f"### {policy_name}'s Value Evaluation: {value}\n")


        environment.action(column=int(agent_move.item()))
        environment.player = -1
        graphic(environment.state)
        print("\n")
        proceed = True

    else:
        print("❌ ERROR: invalid colour\n")


while (column != "end"):

    proceed = False
    inrange = False
    
    while not proceed:
        column = input(f"{emoji} Select any column from 1 to 7: ")
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
        environment.action(column=int(column)-1)
        environment.player = agent_colour
        move += 1

        if winner(environment.state) == 1:
            print("\n🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
            graphic(environment.state)
            column = "end"
        
        if winner(environment.state) == -1:
            print("🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
            column = "end"

        print("\n")

        if column == "end":
            break

        else:
            graphic(environment.state)
        

        distribution = tree_search.run(state=environment.state, player=agent_colour, display=False)
        print(f"\n### {policy_name}'s TREE Decision Tensor: {distribution}\n")

        state = torch.cat(((environment.state).reshape(42), torch.tensor([agent_colour])), dim=0)

        output = policy_network(state)
        neural_distribution = output[:7]
        value = output[7]

        print(f"### {policy_name}'s NEURAL Decision Tensor: {neural_distribution}\n")

        agent_move = torch.argmax(distribution)
        print(f"### {policy_name}'s Value Evaluation: {value}\n")

        environment.action(column=int(agent_move.item()))
        environment.player = player_colour
        graphic(environment.state)
        print("\n")

        if winner(environment.state) == 1:
            print("\n🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
            graphic(environment.state)
            column = "end"
        
        if winner(environment.state) == -1:
            print("🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
            column = "end"

