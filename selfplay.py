import torch
from neural_network import policy
from selfplay_functions import parallel_selfplay
from connect_4 import graphic

# Import Model, [42, 64, 64, 64, 8]
torch.serialization.add_safe_globals([policy])
policy_network = torch.load("1000itr lr=0.1 expc=1.85.pt", weights_only=False)

cpu_cores = 8
game_sets = 125

# Hyperparameters
exploration_constant = 1.85
loss_value_constant = 1
loss_dist_constant = 1
epochs = 3
nabla = 0.1
noise = 0

if __name__ == "__main__":

    for _ in range(game_sets):
        X, Y = parallel_selfplay(policy_network=policy_network,
                                 exploration_constant=exploration_constant,
                                 cpu_cores=cpu_cores,
                                 noise=noise)

        for index, state in enumerate(X):
            print(state.reshape(6,7))
            print("")
            print(Y[index])
            print("")

        policy_network.train(training_inputs=X,
                             training_outputs=Y,
                             loss_value_constant=loss_value_constant,
                             loss_dist_constant=loss_dist_constant,
                             epochs=epochs,
                             nabla=nabla)

        torch.save(policy_network, "2000itr lr=0.1 expc=1.85.pt")
