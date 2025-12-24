import torch
from neural_network import policy, Constant_Network
from connect_4 import Grid, winner, graphic, compute_player
from tree_module import MCTS
from functions import alphazero_display, softmax_temp

torch.serialization.add_safe_globals([policy])
policy_network = torch.load("awesome_parameters.pt", weights_only=False)
policy_name = "awesome_parameters.pt"

initial = torch.zeros(6,7)
environment = Grid(state=initial)

column = ""
move = 0
exploration_constant = 1.85
search_depth = 1000
noise = 0

print("""\n# Connect 4. ("end" to exit) +\n""")

proceed = False
while not proceed:
    player_colour = input("🔴 / 🟡 ? : ")
    print("")

    if player_colour == "end":
            column = "end"
            break
    
    if player_colour in {"red", "Red", "1", "r"}:
        emoji = "🔴"
        player_colour = 1
        agent_colour = -1
        graphic(environment.state)
        print("")
        proceed = True

    elif player_colour in {"yellow", "Yellow", "2", "y"}:
        emoji = "🟡"
        player_colour = -1
        agent_colour = 1
   
        tree_search = MCTS(model=policy_network, iterations=search_depth)
        distribution = tree_search.run(state=environment.state, exploration_constant=exploration_constant, display=False)

        state = environment.state.reshape(42)
        output = policy_network(state)
        neural_distribution = output[:7]
        value = output[7]
        root_node = tree_search.explored_nodes[0]
        root_value = root_node.value_sum / root_node.visit_count

        choose_from = softmax_temp(distribution, temp=noise)[0]
        agent_move = torch.multinomial(choose_from, num_samples=1)

        # Display Network Heads
        alphazero_display(policy_name=policy_name,
                          state=environment.state,
                          tree_dist=distribution,
                          tree_value=root_value,
                          neural_dist=neural_distribution,
                          neural_value=value,
                          agent_move=agent_move,
                          noise=noise)

        environment.action(column=int(agent_move.item()))
        graphic(environment.state)
        print("")
        proceed = True

    else:
        print("❌ ERROR: invalid colour\n")


while (column != "end"):

    proceed = False
    inrange = False
    
    while not proceed:
        column = input(f"{emoji} Select Column (1-7) : ")
        print(f"\n+ ---------------------- ( Move {move} )")
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
            player = compute_player(environment.state)
            if player == 1:
                print("🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
                graphic(environment.state)
                column = "end"
            else:
                print("\n🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
                graphic(environment.state)
                column = "end"

        print("")

        if column == "end":
            break

        else:
            graphic(environment.state)
        
        tree_search = MCTS(model=policy_network, iterations=search_depth)
        distribution = tree_search.run(state=environment.state, exploration_constant=exploration_constant, display=False)
        
        flat_state = environment.state.reshape(42)
        output = policy_network.forward(flat_state)
        print(flat_state)
        print("recomp??")
        print(output)
        neural_distribution = output[:7]
        value = output[7]
        root_node = tree_search.explored_nodes[0]
        root_value = root_node.value_sum / root_node.visit_count

        choose_from = softmax_temp(distribution, temp=noise)[0]
        agent_move = torch.multinomial(choose_from, num_samples=1)

        # Display Network Heads
        alphazero_display(policy_name=policy_name,
                          state=environment.state,
                          tree_dist=distribution,
                          tree_value=root_value,
                          neural_dist=neural_distribution,
                          neural_value=value,
                          agent_move=agent_move,
                          noise=noise)

        environment.action(column=int(agent_move.item()))
        graphic(environment.state)
        print("\n")

        if winner(environment.state) == 1:
            player = compute_player(environment.state)
            if player == 1:
                print("🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
                graphic(environment.state)
                column = "end"
            else:
                print("\n🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
                graphic(environment.state)
                column = "end"

