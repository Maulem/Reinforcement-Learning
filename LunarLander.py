import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from DDQLPytorch import DoubleDeepQLearning, renderLunarLander, multipleTestLunarLander, MemoryThread
from multiprocessing import Process, Pipe
from statistics import mean

gamma = 0.99 
epsilon = 0.9
epsilon_min = 0.01
epsilon_dec = 1000
episodes = 2000
batch_size = 128
tau = 0.005         # TAU is the update rate of the target network
lr = 1e-4           # LR is the learning rate of the ``AdamW`` optimizer

def trainLunarLander(arqName, conn):
    newEnv = gym.make('LunarLander-v2')
    print("hello from: " + arqName)
    DDQL = DoubleDeepQLearning(newEnv, gamma, epsilon, epsilon_min, epsilon_dec, batch_size, tau, lr, episodes, arqName)
    accuracy = DDQL.train()
    conn.send(accuracy)
    conn.close()

if __name__ == "__main__":

    #env = gym.make('LunarLander-v2',render_mode="human")
    env = gym.make('LunarLander-v2')
    #np.random.seed(0)

    print('State space: ', env.observation_space)
    print('Action space: ', env.action_space)

    # train = False
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
                arqName = "data/lunarLander/lunarLander" + str(index)
                x = Process(target=trainLunarLander, args=(arqName, child_conn,))
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

        fileName = "data/lunarLander/lunarLander0Policy"

        threadTest = MemoryThread(target = renderLunarLander, args = (fileName, ))
        threadTest.start()

        rewards, goodCount = multipleTestLunarLander(times, 100, verbose = True, fileName = fileName)

        print("Rewards mean: {} points".format(mean(rewards)))
        print("Good rewards rate: {0}/{1} or {2}%".format(goodCount, times, goodCount/times * 100))

    else:
        print("Insert the number of times to test")
        times = int(input())

        fileName = "data/lunarLander/bestPolicy"

        threadTest = MemoryThread(target = renderLunarLander, args = (fileName, ))
        threadTest.start()

        rewards, goodCount = multipleTestLunarLander(times, 100, verbose = True, fileName = fileName)

        print("Rewards mean: {} points".format(mean(rewards)))
        print("Good rewards rate: {0}/{1} or {2}%".format(goodCount, times, goodCount/times * 100))

