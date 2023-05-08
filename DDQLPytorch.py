import gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import math
import random
from collections import namedtuple, deque
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
import statistics
import threading

###| DEFINES |###
    
TRANSITION = namedtuple('Transition', ('state', 'action', 'next_state', 'reward') )

DEVICE = None

IS_IPYTHON = 'inline' in matplotlib.get_backend()

if IS_IPYTHON:
    from IPython import display

killSwitch = False

###| CORE |###

# Thread that can save the return of the function in self.value
class MemoryThread(threading.Thread):
    # constructor
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        # execute the base constructor
        threading.Thread.__init__(self, group, target, name,
                 args, kwargs)
        # set a default value
        self.value = None
 
    def run(self):
        """Method representing the thread's activity.

        You may override this method in a subclass. The standard run() method
        invokes the callable object passed to the object's constructor as the
        target argument, if any, with sequential and keyword arguments taken
        from the args and kwargs arguments, respectively.

        """
        try:
            if self._target:
                self.value = self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs
    
class DoubleDeepQLearning():

    class ReplayMemory(object):

        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(TRANSITION(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)
        
    class DQN(nn.Module):

        def __init__(self, n_observations, n_actions):
            super(DoubleDeepQLearning.DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 512)
            self.layer2 = nn.Linear(512, 256)
            self.layer3 = nn.Linear(256, 128)
            self.layer4 = nn.Linear(128, 64)
            self.layer5 = nn.Linear(64, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            x = F.relu(self.layer4(x))
            return self.layer5(x)
    
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, batch_size, tau, lr, num_episodes, arqName):

        # Turn plt interactive mode on
        plt.ion()

        self.env = env
        self.gamma = gamma              # gamma is the discount factor as mentioned in the previous section
        self.epsilon = epsilon          # epsilon is the starting value of epsilon
        self.epsilon_min = epsilon_min  # epsilon_min is the final value of epsilon
        self.epsilon_dec = epsilon_dec  # epsilon_dec controls the rate of exponential decay of epsilon, higher means a slower decay
        self.batch_size = batch_size    # batch_size is the number of transitions sampled from the replay buffer
        self.tau = tau                  # tau is the update rate of the target network
        self.lr = lr                    # lr is the learning rate of the ``AdamW`` optimizer
        self.num_episodes = num_episodes
        self.arqName = arqName
        self.episode_durations = []
        self.rewards = []
        self.steps_done = 0

        # Set the memory
        self.memory = self.ReplayMemory(10000)

        # Get number of actions from gym action space
        n_actions = self.env.action_space.n

        # Get the number of state observations
        state, info = self.env.reset()
        n_observations = len(state)

        global DEVICE
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")

        # Create the policy neural network
        self.policy_net = self.DQN(n_observations, n_actions).to(DEVICE)

        # Create and copy the target neural network from the policy neural network
        self.target_net = self.DQN(n_observations, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Set the optimizer 
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.lr, amsgrad=True)

        self.durationMeans = []
        self.rewardMeans = []


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            math.exp(-1. * self.steps_done / self.epsilon_dec)
        self.steps_done += 1

        # If true returns the neural network response, else random value
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device = DEVICE, dtype=torch.long)
        
    def onTimePlot(self, show_result = False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy(), label="Duration")
        plt.plot(rewards_t.numpy(), label="Reward")
        # Take 100 episode averages and plot them too

        means = rewards_t.mean().view(-1)
        self.rewardMeans.append(means.numpy())
        plt.plot(self.rewardMeans, label="Rewards mean")
        
        means = durations_t.mean().view(-1)
        self.durationMeans.append(means.numpy())
        plt.plot(self.durationMeans, label="Durations mean")
        
        
        plt.legend()

        plt.pause(0.001)  # pause a bit so that plots are updated
        if IS_IPYTHON:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = TRANSITION(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device = DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device = DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):

        if DEVICE == torch.device("cuda"):
            print("Using GPU!")
        else:
            print("Using CPU!")

        threadTest = None
        goodEpsCount = 0
        global killSwitch

        for i_episode in range(self.num_episodes):
            # Initialize the environment and get it's state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device = DEVICE).unsqueeze(0)
            rewards = 0
            
        
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device = DEVICE)
                done = terminated or truncated or t > 1000

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device = DEVICE).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                rewards += reward[0].item()

                if done:
                    self.episode_durations.append(t + 1)
                    self.rewards.append(rewards)
                    self.onTimePlot()
                    break

            if t + 1 >= 500:
                goodEpsCount += 1
            else:
                goodEpsCount = 0

            if i_episode % 50 == 0:
                torch.save(self.policy_net.state_dict(), self.arqName + "Policy")
                torch.save(self.target_net.state_dict(), self.arqName + "Target")
                if self.env.spec.id == "LunarLander-v2":
                    if threadTest is not None:
                        if threadTest.is_alive():
                            killSwitch = True
                            threadTest.join()
                    threadTest = MemoryThread(target = renderLunarLander, args = (self.arqName + "Policy", ))
                    threadTest.start()
                    testTimes = 100
                    threads =  100
                    _rewards, goodCount = multipleTestLunarLander(testTimes, threads, fileName = self.arqName + "Policy")
                    accuracy = goodCount/testTimes * 100
                    print("Tested accuracy from {0} is {1}%".format(self.arqName, accuracy))
                    if goodCount > int(0.98 * testTimes): break
                elif self.env.spec.id == "CartPole-v1":
                    if threadTest is not None:
                        if threadTest.is_alive():
                            killSwitch = True
                            threadTest.join()
                    threadTest = MemoryThread(target = renderCartpole, args = (self.arqName + "Policy", ))
                    threadTest.start()
                    accuracy = 100
                    if goodEpsCount >= 50: break


        print("Complete {0} with {1}% accuracy".format(self.arqName, accuracy))
        self.onTimePlot(show_result=True)
        plt.ioff()

        torch.save(self.policy_net.state_dict(), self.arqName + "Policy")
        torch.save(self.target_net.state_dict(), self.arqName + "Target")

        if threadTest is not None:
            if threadTest.is_alive():
                killSwitch = True
                threadTest.join()
        
        if self.env.spec.id == "LunarLander-v2":
            plt.savefig("results/lunarLander/DDQL.png")
        elif self.env.spec.id == "CartPole-v1":
            plt.savefig("results/cartPole/DDQL.png")
        return accuracy
    
def renderCartpole(path = "data/model_test_policy"):

    global killSwitch

    def nextAction(state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1).item()
        
    # Create the enviroment in human render mode
    env = gym.make('CartPole-v1', render_mode="human")

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Get the number of state observations
    observation, info = env.reset()
    n_observations = len(observation)

    # If GPU is to be used
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loads the neural network to test
    policy_net = DoubleDeepQLearning.DQN(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(torch.load(path))

    # Returns an initial state
    state = torch.tensor(observation, dtype=torch.float32, device = DEVICE).unsqueeze(0)

    # Starts the loop
    done = False

    startTime = time.time()
    lastTime = -1

    while not done:

        env.render()

        # The neural network produces either 0 (left) or 1 (right).
        observation, reward, terminated, truncated, _ = env.step(nextAction(state))

        if terminated:
            state = None
        else:
            state = torch.tensor(observation, dtype=torch.float32, device = DEVICE).unsqueeze(0)

        nowTime = time.time()

        if int(nowTime - startTime) > lastTime:
            sys.stdout.write("                                 " + '\r')
            sys.stdout.flush()
            sys.stdout.write("O cartpole esta em pé a {} segundos.".format(int(nowTime - startTime)))
            sys.stdout.flush()
            lastTime = int(nowTime - startTime)

        if terminated or killSwitch:
            killSwitch = False
            done = True

    env.close()

def renderLunarLander(path = "data/model_test_policy"):
    def nextAction(state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1).item()

    env = gym.make('LunarLander-v2', render_mode='human').env

    global killSwitch

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Get the number of state observations
    observation, info = env.reset()
    n_observations = len(observation)

    # Loads the neural network to test
    policy_net = DoubleDeepQLearning.DQN(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(torch.load(path))

    # Returns an initial state
    state = torch.tensor(observation, dtype=torch.float32, device = DEVICE).unsqueeze(0)

    # Starts the loop
    done = False

    rewards = 0
    steps = 0
    max_steps = 500

    while (not done) and (steps < max_steps):

        # The neural network produces either 0 (left) or 1 (right).
        observation, reward, terminated, truncated, _ = env.step(nextAction(state))

        if terminated:
            state = None
        else:
            state = torch.tensor(observation, dtype=torch.float32, device = DEVICE).unsqueeze(0)


        rewards += reward
        env.render()
        steps += 1

        if terminated or killSwitch:
            killSwitch = False
            done = True



    print(f'Score = {rewards}')

    env.close()


def testLunarLander(fileName = "data/lunarLander/98%_policy"):
    def nextAction(state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1).item()

    env = gym.make('LunarLander-v2').env

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Get the number of state observations
    observation, info = env.reset()
    n_observations = len(observation)

    # Loads the neural network to test
    policy_net = DoubleDeepQLearning.DQN(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(torch.load(fileName))

    # Returns an initial state
    state = torch.tensor(observation, dtype=torch.float32, device = DEVICE).unsqueeze(0)

    # Starts the loop
    done = False

    rewards = 0
    steps = 0
    max_steps = 500

    while (not done) and (steps < max_steps):

        # The neural network produces either 0 (left) or 1 (right).
        observation, reward, terminated, truncated, _ = env.step(nextAction(state))

        if terminated:
            state = None
        else:
            state = torch.tensor(observation, dtype=torch.float32, device = DEVICE).unsqueeze(0)


        rewards += reward

        steps += 1

        if terminated: 
            done = True

    env.close()

    return rewards

def multipleTestLunarLander(times, maxThreads, verbose = False, fileName = "data/lunarLander/98%_policy"):
    rewards = []
    goodCount = 0

    if times < maxThreads:
        maxThreads = times
    
    for _ in range(int(times / maxThreads)):
        threads = list()
        for index in range(maxThreads):
            x = MemoryThread(target=testLunarLander, args=(fileName, ))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()
            rewards.append(thread.value)
            if verbose:
                print(f'Score = {thread.value}')
            if thread.value >= 200:
                goodCount += 1
    return rewards, goodCount