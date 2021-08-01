import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

#Hyperparameters
learning_rate = 0.001
gamma = 0.99
seed = 13
epsilon = np.finfo(np.float32).eps
epochs = 2000


#Initializing environment
env = gym.make('CartPole-v1')

#Setting up the seeds
env.seed(seed)
torch.manual_seed(seed)


class PolicyNetwork(nn.Module):

    def __init__(self,environment):
        super(PolicyNetwork,self).__init__()

        self.state_space = environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.fc1 = nn.Linear(self.state_space, 128)
        self.fc2 = nn.Linear(128, self.action_space)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = []
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


def select_action(state,network):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = network(state)
    m = Categorical(probs)
    action = m.sample()
    network.policy_history.append(m.log_prob(action))
    return action.item()



def update_policy(network,optimizer):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in network.reward_episode[::-1]:
        R = r + network.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)

    # Calculate loss
    network.policy_history = torch.cat(network.policy_history)
    loss = network.policy_history*rewards
    loss = -loss
    #loss = (torch.sum(torch.mul(network.policy_history, rewards).mul(-1), -1))
    loss = loss.sum()

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    network.loss_history.append(loss.item())
    network.reward_history.append(sum(network.reward_episode))
    network.policy_history = []
    network.reward_episode = []




if __name__ == "__main__":

    #Initializing the network and optimizer
    net = PolicyNetwork(env)
    opt = optim.Adam(net.parameters(),lr=learning_rate)


    running_reward = 10
    average_reward=0
    iter = 0
    for episode in range(epochs):
        state = env.reset()  # Reset environment and record the starting state
        done = False

        for time in range(10000):
            action = select_action(state,net)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action)
            # Save reward
            net.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)
        episode_total_reward = sum(net.reward_episode)
        average_reward = 0.05 * episode_total_reward + (1 - 0.05) * average_reward
        update_policy(net,opt)

        if episode % 50 == 0:
            print('Episode {}\tLast length:{:5d}\tAverage length:{:.2f}\tEpisode Reward:{:.2f}\tAverage Reward:{:.2f}'.format(episode, time, running_reward,episode_total_reward,average_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break


    x = np.array([i for i in range(len(net.reward_history))])
    r = np.array(net.reward_history)
    l = np.array(net.loss_history)

    plt.plot(x,r)
    plt.xlabel("Iterations")
    plt.ylabel("Iteration Reward")
    plt.show()

    plt.plot(x, l)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


