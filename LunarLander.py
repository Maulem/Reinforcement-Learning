import gym
from DDQLPytorch import DoubleDeepQLearning
from multiprocessing import Process, Pipe
from statistics import mean
from NeuralNetworks import LunarLanderNeuralNetwork
from DDQLTests import renderLunarLander, multipleTestLunarLander, MemoryThread

###|CONSTANTS|###

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

def trainLunarLander(LunarLanderNetwork, fileName, conn):
    print("hello from: " + fileName)
    env = gym.make('LunarLander-v2')
    DDQL = DoubleDeepQLearning( env, LunarLanderNetwork, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DEC,
                                BATCH_SIZE, TAU, LR, NUM_EPISODES, STOP, fileName)
    accuracy = DDQL.train()
    conn.send(accuracy)
    conn.close()

if __name__ == "__main__":
    
    LunarLanderNetwork = LunarLanderNeuralNetwork

    print("Choose which mode to run:")
    print("1 - Train the model")
    print("3 - Test the last model performance")
    print("3 - Test the best model performance")
    train = int(input())

    bestAccuracy = 0

    if train == 1:

        while True:
            processes = list()

            for index in range(2):
                parent_conn, child_conn = Pipe()
                arqName = FILENAME + str(index)
                x = Process(target = trainLunarLander, args=(LunarLanderNetwork, arqName, child_conn, ))
                processes.append((x, parent_conn))
                x.start()

            for index, processItem in enumerate(processes):
                process = processItem[0]
                parent_conn = processItem[1]
                accuracy = parent_conn.recv()
                process.join()
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                print(accuracy)
                if accuracy >= 100:
                    break
            if accuracy >= 100:
                for index, processItem in enumerate(processes):
                    processItem[0].terminate()
                break


    elif train == 2:
        print("Insert the number of times to test")
        times = int(input())

        fileName = FILENAME + "0Policy"

        threadTest = MemoryThread(target = renderLunarLander, args = (LunarLanderNetwork, fileName, ))
        threadTest.start()

        rewards, goodCount = multipleTestLunarLander(times, MAX_THREADS, LunarLanderNetwork, fileName, verbose = True)

        print("Rewards mean: {} points".format(mean(rewards)))
        print("Good rewards rate: {0}/{1} or {2}%".format(goodCount, times, goodCount/times * 100))

    else:
        print("Insert the number of times to test")
        times = int(input())

        fileName = "data/lunarLander/bestPolicy"

        threadTest = MemoryThread(target = renderLunarLander, args = (LunarLanderNetwork, fileName, ))
        threadTest.start()

        rewards, goodCount = multipleTestLunarLander(times, MAX_THREADS, LunarLanderNetwork, fileName, verbose = True)

        print("Rewards mean: {} points".format(mean(rewards)))
        print("Good rewards rate: {0}/{1} or {2}%".format(goodCount, times, goodCount/times * 100))

