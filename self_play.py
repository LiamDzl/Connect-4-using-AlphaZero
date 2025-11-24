import torch
from neural_network import policy
from connect_4 import mask, graphic, winner, Grid
from tree_module import PUCT, Node, MCTS
from functions import colour

reward_boost = 0.8

policy_network = policy(structure=[43,100,100,42,8], alpha=reward_boost)

tree_test = MCTS(model=policy_network, iterations=600)
    
print(f"\n### Self Play Interface ###\n")
print("---------------------------------------------------------------------------------------------")

game_state = torch.zeros(6,7)
player = 1
colour_index = 1
while winner(game_state) == 0 and game_state.nonzero != 42:

    print(f"\n### {colour(colour_index)}'s Move ###\n")
    graphic(game_state)
    print("\n")

    mcts_distribution = tree_test.run(state=game_state,
                                    player=player,
                                    display=False)
    
    print(f"### {colour(colour_index)}'s Decision Tensor: {mcts_distribution}\n")

    print("---------------------------------------------------------------------------------------------")

    chosen_move = torch.multinomial(mcts_distribution, num_samples=1).item()
    grid = Grid(state=game_state, player=player)
    grid.action(chosen_move)
    # New state
    game_state = grid.state
    player = player * -1
    colour_index *= -1

if winner(game_state) == 0:
    print(f"\n### Final Drawn Position ###\n")

if winner(game_state) == 1:
    print(f"\n### Final Position: Red Win ###\n")

if winner(game_state) == -1:
    print(f"\n### Final Position: Yellow Win ###\n")

graphic(game_state)
print("\n")