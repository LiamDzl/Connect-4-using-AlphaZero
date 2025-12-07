import torch
from neural_network import policy
from selfplay_functions import parallel_selfplay

# Import Model, [43,20,20,20,20,8]
torch.serialization.add_safe_globals([policy])
policy_network = torch.load("weird_parameters.pt", weights_only=False)

for i in range(10):
    if __name__ == "__main__":
        X, Y = parallel_selfplay(policy_network=policy_network)
        policy_network.train(training_inputs=X, training_outputs=Y, epochs=1, nabla=0.04)
        torch.save(policy_network, "weird_parameters.pt")