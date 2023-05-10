import gymnasium as gym
from DDQLPytorch import DoubleDeepQLearning, renderCliffWalking

env = gym.make('CliffWalking-v0')

GAMMA = 0.99        # GAMMA is the discount factor as mentioned in the previous section
EPSILON = 0.9       # EPSILON is the starting value of epsilon
EPSILON_MIN = 0.1  # EPSILON_MIN is the final value of epsilon
EPSILON_DEC = 10000  # EPSILON_DEC controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 128    # BATCH_SIZE is the number of transitions sampled from the replay buffer
TAU = 0.005         # TAU is the update rate of the target network
LR = 1e-4           # LR is the learning rate of the ``AdamW`` optimizer
NUM_EPISODES = 10000
fileName = "data/cartPole/cliff"

print("Choose which mode to run:")
print("1 - Train the model")
print("2 - Render the last model")
print("3 - Render the best model")
train = int(input())
if train == 1:
    DDQL = DoubleDeepQLearning(env, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DEC, BATCH_SIZE, TAU, LR, NUM_EPISODES, fileName, 100)
    DDQL.train()
    renderCliffWalking( fileName + "Policy")

elif train == 2:
    # Render the last model trained by you
    renderCliffWalking( fileName + "Policy")

else:
    # Render the best model trained with 1000 episodes
    renderCliffWalking("data/cartPole/bestPolicy")