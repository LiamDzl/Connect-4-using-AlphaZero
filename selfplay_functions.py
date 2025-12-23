import torch
from neural_network import policy, Constant_Network
from connect_4 import mask, graphic, winner, Grid, compute_player
from tree_module import MCTS
from functions import colour, generate_57, softmax_temp
from multiprocessing import Pool

constant_model = Constant_Network()

def selfplay(policy_network, initial_state, exploration_constant, noise):
    X = [] # Recorded States (Inputs)
    Y = [] # Recorded Distribution/Value Outputs

    tree_test = MCTS(model=policy_network, iterations=250)
    tree_constant = MCTS(model=constant_model, iterations=250)

    game_state = initial_state
    initial_player = compute_player(initial_state)
    player = compute_player(game_state)
    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])
    game_length = 0

    recorded_states = []
    recorded_mcts_dists = []

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):
    
        print(f"\n# {colour(player)}'s Move\n")
        graphic(game_state)
        print("\n")
        
        mcts_distribution = tree_test.run(state=game_state,
                                          exploration_constant=exploration_constant,
                                          display=False)
        
        # Data collection
        x = game_state.reshape(42)
        recorded_states.append(x)
        y = mcts_distribution.reshape(7)
        recorded_mcts_dists.append(y)

        # Choose Move
        choose_from = softmax_temp(mcts_distribution, temp=noise)[0]
        chosen_move = torch.multinomial(choose_from, num_samples=1)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        print(f"# {colour(player)}'s Decision Tensor: {mcts_distribution}\n")
        print(f"# 🌿 Tree w/ Temperature:")
        print(softmax_temp(mcts_distribution, temp=noise))
        print("")
        print(f"# Chosen Move: {chosen_move}")

        # New state
        game_state = grid.state
        player *= -1
        game_length += 1

        # As to end when board full
        filter = mask(game_state)

    if winner(game_state) == 0:
        z = 0

    if winner(game_state) == 1:
        z = compute_player(game_state) * -1 # True Winner, +1 Red, -1 Yellow

    # Concatenate z values to distribution vectors - as to make vectors of length 8 from 7
    dist_value_outputs = []
    z = z * initial_player # if starting player and result match, then reward the initial state... (+1)
    for y in recorded_mcts_dists:
        dist_value_outputs.append(torch.cat((y, torch.tensor([z])), dim=0))
        z *= -1

    graphic(game_state)
    print("\n")

    # Append game results into global dataset, X and Y
    for recorded_state in recorded_states:
        X.append(recorded_state)

    for dist_value_output in dist_value_outputs:
        Y.append(dist_value_output)

    print(Y)
    return X, Y # Training Data

# CPU Parallelisation

def worker(args):
    policy_network, initial_state, exploration_constant, noise = args
    X, Y = selfplay(policy_network=policy_network, initial_state=initial_state, exploration_constant=exploration_constant, noise=noise)
    return X, Y

def parallel_selfplay(policy_network, exploration_constant, noise, cpu_cores):
    initial_states = generate_57()
    move_order = [1, -1, -1, -1, -1, -1 , -1, -1]
    cpu_cores = cpu_cores
    args_list = [(policy_network, state, exploration_constant, noise) for state in initial_states[:cpu_cores]]

    with Pool(cpu_cores) as p:
        dataset = p.map(worker, args_list) # Store 8 Game Results, as pairs (X_1, Y_1, ... X_8, Y_8)

    # Compute Dataset Size:
    dataset_size = 0
    for X, Y in dataset:
        dataset_size += len(X)

    # Compute X, Y Tensors:
    X_total = torch.zeros(dataset_size, 42)
    Y_total = torch.zeros(dataset_size, 8)

    index = 0
    for X, Y in dataset:
        for input, output in zip(X, Y):
            X_total[index] = input
            Y_total[index] = output
            index += 1

    print(f"\n8 Games Complete. New Data Points: {dataset_size}")
    return X_total, Y_total
