import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tree_module import MCTS
from functions import alphazero_display
from connect_4 import mask, graphic

class policy(nn.Module):
    def __init__(self, structure):
        super().__init__()
        layers = [] # this object is a python list, but contains actual affine maps 

        for i in range(len(structure) - 1):  # n layers need n-1 Linear maps
            layers.append(nn.Linear(structure[i], structure[i + 1]))

        self.structure = structure
        self.layers = nn.ModuleList(layers)
        self.pspace_dimensions = []

        for i in range(len(structure) - 1):
            self.pspace_dimensions.append(structure[i] * structure[i+1]) # size of weight matrix
            self.pspace_dimensions.append(structure[i+1]) # size of bias vector

        self.total_dimension = 0
        for i in self.pspace_dimensions:
            self.total_dimension += i

    def forward(self, x):
        state = x.reshape(6,7)

        for i in self.layers[:-1]: # ReLU, up til last (since we wanna grab this vector)
            x = F.relu(i(x))

        penultimate_eight = self.layers[-1](x)
        dist = penultimate_eight[0:7]
        value = penultimate_eight[-1]

        # Force valuation between -1 and 1
        value = 2 * torch.sigmoid(value) - 1

        # Splitting value from distribution + masking
        filter = mask(state)
        to_softmax = dist[filter] #<=7

        reduced_distribution = F.softmax(to_softmax, dim = 0)
        
        distribution = torch.zeros(7)
        distribution[filter] = reduced_distribution
        
        output_vector = torch.cat([distribution, value.unsqueeze(0)], dim=0)
        output_vector = output_vector.reshape(8)

        return output_vector
    
    def train(self, training_inputs, training_outputs, loss_value_constant, loss_dist_constant, epochs, nabla):
            training_size = training_inputs.shape[0]
            mse = nn.MSELoss()
            optimiser = optim.SGD(self.parameters(), lr=nabla)

            for epoch in range(epochs):
                permutation = torch.randperm(training_size)
                training_inputs = training_inputs[permutation]
                training_outputs = training_outputs[permutation]

                for x, y in zip(training_inputs, training_outputs):
                    nn_output = self.forward(x)

                    # Value and policy
                    value_loss = mse(nn_output[7], y[7])
                    policy_loss = -torch.dot(y[0:7], torch.log(nn_output[0:7] + 1e-8))

                    loss_scalar = (loss_value_constant) * value_loss + (loss_dist_constant) * policy_loss

                    optimiser.zero_grad()
                    loss_scalar.backward()
                    optimiser.step()

            return "Successfully Trained."
    
    def evaluate(self, name, state, exploration_constant, noise):
        print("\n")
        graphic(state)
        print("\n")
        state = state.reshape(42)
        state = state.float()
        output = self.forward(state)
        neural_distribution = output[:7]
        value = output[7]

        tree_search = MCTS(model=self, iterations=100)
        state = state.reshape(6, 7)
        dist = tree_search.run(state=state, exploration_constant=exploration_constant, display=False)
        root_node = tree_search.explored_nodes[0]
        root_value = root_node.value_sum / root_node.visit_count

        alphazero_display(policy_name=name,
                          state=state,
                          tree_dist=dist,
                          tree_value=root_value,
                          neural_dist=neural_distribution,
                          neural_value=value,
                          agent_move=None,
                          noise=noise)

    
# Default (CONSTANT) Network for Testing

class Constant_Network:
    def __init__(self):
        pass

    def forward(self, x):
        normal = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        state = x.reshape(6,7)
        filter = mask(state)
        normal_filtered = normal[filter]
        normal_filtered = torch.softmax(normal_filtered, dim=0)
        distribution = torch.zeros(7)
        distribution[filter] = normal_filtered
        output = torch.cat((distribution, torch.tensor([0])))
        return output
