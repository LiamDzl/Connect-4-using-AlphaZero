import torch
from neural_network import policy
from connect_4 import mask, graphic, winner, Grid
from tree_module import PUCT, Node, MCTS
from functions import colour

# structure is [43,20,20,20,20,8]

# Import Models
torch.serialization.add_safe_globals([policy])
policy_network = torch.load("weird_parameters.pt", weights_only=False)
tree_test = MCTS(model=policy_network, iterations=600)

games = 20
    
print(f"\n### Self Play Interface ###\n")
print("---------------------------------------------------------------------------------------------")

for game in range(games):
    print(f"\n# # # # # # GAME {game} # # # # # #\n")
    game_state = torch.zeros(6,7)
    player = 1
    colour_index = 1
    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])
    game_length = 0

    recorded_states = []
    recorded_mcts_dists = []

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):
    
        print(f"\n### {colour(colour_index)}'s Move ###\n")
        graphic(game_state)
        print("\n")

        mcts_distribution = tree_test.run(state=game_state,
                                        
                                        player=player,
                                        display=False)
        
        # Data collection
        x = torch.cat((game_state.reshape(42), torch.tensor([player])), dim=0)
        recorded_states.append(x)
        y = mcts_distribution.reshape(7)
        recorded_mcts_dists.append(y)
        
        print(f"### {colour(colour_index)}'s Decision Tensor: {mcts_distribution}\n")

        print("---------------------------------------------------------------------------------------------")

        chosen_move = torch.argmax(mcts_distribution, dim=1)
        grid = Grid(state=game_state, player=player)
        grid.action(chosen_move)

        # New state
        game_state = grid.state
        player = player * -1
        colour_index *= -1

        game_length += 1

        # As to end when board full
        filter = mask(game_state)

    if winner(game_state) == 0:
        print(f"\n### Final Drawn Position ###\n")
        z = 0

    if winner(game_state) == 1:
        print(f"\n### Final Position: Red Win ###\n")
        z = 1

    if winner(game_state) == -1:
        print(f"\n### Final Position: Yellow Win ###\n")
        z = -1

    # Add Z scores
    dist_results = []
    for y in recorded_mcts_dists:
        dist_results.append(torch.cat((y, torch.tensor([z])), dim=0))

    graphic(game_state)
    print("\n")


    X = torch.zeros(game_length, 43)
    for index in range(game_length):
        X[index] = recorded_states[index]

    print(X)

    Y = torch.zeros(game_length, 8)
    for index in range(game_length):
        Y[index] = dist_results[index]

    print(Y)

    policy_network.train(training_inputs=X, training_outputs=Y, epochs=1, nabla=0.004)

torch.save(policy_network, "weird_parameters.pt")
