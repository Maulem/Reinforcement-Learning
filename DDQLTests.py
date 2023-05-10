import torch
import gym
import time
import sys
import threading

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


killSwitch = False


def renderCartpole(neuralNetwork, neuralNetworkFile):

    global killSwitch

    def nextAction(state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1).item()
        
    # Create the enviroment in human render mode
    env = gym.make('CartPole-v1', render_mode='human')

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Get the number of state observations
    observation, info = env.reset()
    n_observations = len(observation)

    # If GPU is to be used
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loads the neural network to test
    policy_net = neuralNetwork(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(torch.load(neuralNetworkFile))

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


def renderLunarLander(neuralNetwork, neuralNetworkFile):
    def nextAction(state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1).item()

    env = gym.make('LunarLander-v2', render_mode='human').env

    global killSwitch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Get the number of state observations
    observation, info = env.reset()
    n_observations = len(observation)

    # Loads the neural network to test
    policy_net = neuralNetwork(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(torch.load(neuralNetworkFile))

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


def testLunarLander(neuralNetwork, neuralNetworkFile):
    def nextAction(state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1).item()

    env = gym.make('LunarLander-v2').env

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Get the number of state observations
    observation, info = env.reset()
    n_observations = len(observation)

    # Loads the neural network to test
    policy_net = neuralNetwork(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(torch.load(neuralNetworkFile))

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

def multipleTestLunarLander(times, maxThreads, neuralNetwork, neuralNetworkFile, verbose = False):
    rewards = []
    goodCount = 0

    if times < maxThreads:
        maxThreads = times
    
    for _ in range(int(times / maxThreads)):
        threads = list()
        for index in range(maxThreads):
            x = MemoryThread(target=testLunarLander, args=(neuralNetwork, neuralNetworkFile, ))
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