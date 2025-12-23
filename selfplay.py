import torch
from neural_network import policy
from selfplay_functions import parallel_selfplay
from connect_4 import graphic
import copy

# Import Model, [42, 64, 64, 64, 8]
torch.serialization.add_safe_globals([policy])
policy_network = torch.load("1000itr lr=0.001 expc=1.85.pt", weights_only=False)

cpu_cores = 8
game_sets = 125

# Hyperparameters
exploration_constant = 1.35
loss_value_constant = 1
loss_dist_constant = 1
epochs = 3
nabla = 0.001
noise = 0

if __name__ == "__main__":

    for _ in range(game_sets):
        X, Y = parallel_selfplay(policy_network=policy_network,
                                 exploration_constant=exploration_constant,
                                 cpu_cores=cpu_cores,
                                 noise=noise)

        X_original = X.clone()
        Y_original = Y.clone()

        # Augment Data - Use Symmetry of Connect 4
        X_mirror = X.clone()
        Y_mirror = Y.clone()

        for index, state in enumerate(X_original):
            shaped = state.reshape(6,7)
            print(shaped)
            shaped_mirror = torch.flip(shaped, dims=[1])
            X_mirror[index] = shaped_mirror.reshape(42)
            print("")
            print(shaped_mirror)
            print("")
            y = Y_original[index]
            y_dist = y[:7]
            y_value = y[7]
            y_mirror = torch.cat((torch.flip(y_dist, dims=[0]), torch.tensor([y_value])), dim=0)
            Y_mirror[index] = y_mirror
            print(y)
            print(y_mirror)
            print("")

        X_symm = torch.cat((X_original, X_mirror), dim=0)
        Y_symm = torch.cat((Y_original, Y_mirror), dim=0)

        policy_network.train(training_inputs=X_symm,
                             training_outputs=Y_symm,
                             loss_value_constant=loss_value_constant,
                             loss_dist_constant=loss_dist_constant,
                             epochs=epochs,
                             nabla=nabla)

        torch.save(policy_network, "1000itr lr=0.001 expc=1.85.pt")
