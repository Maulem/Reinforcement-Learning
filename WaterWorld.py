from pettingzoo.sisl import waterworld_v4
from DDQLPytorch import DoubleDeepQLearningMultiAgent
from NeuralNetworks import WaterWorldNeuralNetwork

env = waterworld_v4.env(n_pursuers=2, n_evaders=2, n_poisons=10, render_mode='human')
env.reset()

GAMMA = 0.99        # GAMMA is the discount factor as mentioned in the previous section
EPSILON = 0.9       # EPSILON is the starting value of epsilon
EPSILON_MIN = 0.01  # EPSILON_MIN is the final value of epsilon
EPSILON_DEC = 1000  # EPSILON_DEC controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 128    # BATCH_SIZE is the number of transitions sampled from the replay buffer
TAU = 0.005         # TAU is the update rate of the target network
LR = 1e-4           # LR is the learning rate of the ``AdamW`` optimizer
NUM_EPISODES = 2000
STOP = 0.98
FILENAME = "data/lunarLander/lunarLander"
MAX_THREADS = 100

DDQL = DoubleDeepQLearningMultiAgent(  env, WaterWorldNeuralNetwork, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DEC,
                                BATCH_SIZE, TAU, LR, NUM_EPISODES, STOP, FILENAME)

DDQL.train()

if __name__ == "__main__":

    env.reset()
    for agent in env.agent_iter():
        env.render()
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
            action = [0, 0]
            print(action)
        env.step(action)
    env.close()