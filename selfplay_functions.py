import torch
from neural_network import policy
from connect_4 import mask, graphic, winner, Grid
from tree_module import MCTS
from functions import colour, generate_57
from multiprocessing import Pool


def selfplay(policy_network, initial_state, move_first):
    X = [] # Recorded States (Inputs)
    Y = [] # Recorded Distribution/Value Outputs

    tree_test = MCTS(model=policy_network, iterations=200)

    game_state = initial_state
    player = move_first
    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])
    game_length = 0

    recorded_states = []
    recorded_mcts_dists = []

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):
    
        print(f"\n### {colour(player)}'s Move ###\n")
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
        
        print(f"### {colour(player)}'s Decision Tensor: {mcts_distribution}\n")

        print("---------------------------------------------------------------------------------------------")

        chosen_move = torch.argmax(mcts_distribution, dim=1)
        grid = Grid(state=game_state, player=player)
        grid.action(chosen_move)

        # New state
        game_state = grid.state
        player *= -1
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

    # Concatenate z values to distribution vectors - as to make vectors of length 8 from 7
    dist_value_outputs = []
    for y in recorded_mcts_dists:
        dist_value_outputs.append(torch.cat((y, torch.tensor([z])), dim=0))

    graphic(game_state)
    print("\n")

    # Append game results into global dataset, X and Y
    for recorded_state in recorded_states:
        X.append(recorded_state)

    for dist_value_output in dist_value_outputs:
        Y.append(dist_value_output)


    return X, Y # Training Data

# --- CPU Parallelisation --- #

def worker(args):
    policy_network, initial_state, move_first = args
    X, Y = selfplay(policy_network=policy_network, initial_state=initial_state, move_first=move_first)
    return X, Y

def parallel_selfplay(policy_network):
    initial_states = generate_57()
    move_order = [1, -1, -1, -1, -1, -1 , -1, -1]
    cpu_cores = 8
    args_list = [(policy_network, state, to_move) for state, to_move in zip(initial_states[:8], move_order)]

    with Pool(cpu_cores) as p:
        dataset = p.map(worker, args_list) # Store 8 Game Results, as pairs (X_1, Y_1, ... X_1, Y_1)

    # Compute Dataset Size:
    dataset_size = 0
    for X, Y in dataset:
        dataset_size += len(X)

    # Compute X, Y Tensors:
    X_total = torch.zeros(dataset_size, 43)
    Y_total = torch.zeros(dataset_size, 8)

    index = 0
    for X, Y in dataset:
        for input, output in zip(X, Y):
            X_total[index] = input
            Y_total[index] = output
            index += 1

    print(f"\n8 Games Complete. New Data Points: {dataset_size}")
    return X_total, Y_total