import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from Datagen import generate_geometry_CSI,generate_rayleigh_CSI,open_dataset
from random import random


#Hyperparameters
learning_rate = 0.01
seed = 5
epsilon = np.finfo(np.float32).eps
epochs = 3000

num_users = 4
power_levels = 5
power_unit = float(1.0/float(power_levels))
noise_ratio = 0.1
batch_size = 32

PATH=" model"
BESTPATH = "bestModel"

#Setting up the seeds
torch.manual_seed(seed)
np.random.seed(seed)
rng = np.random.RandomState(1)
zeta = 1


class PolicyNetwork(nn.Module):

    def __init__(self,num_users,power_levels):
        super(PolicyNetwork,self).__init__()

        self.num_users = num_users
        self.power_levels = power_levels

        self.fc1 = nn.Linear(self.num_users*self.num_users, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output = []

        for i in range(self.num_users):
            self.output.append(nn.Linear(256,self.power_levels))


        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        ret = []
        for i in range(self.num_users):
            ret.append(torch.softmax(self.output[i](x),dim=1))
        return ret



def create_power_vector(input,network):
    input = torch.from_numpy(input).float()
    output = network(input)

    #print("Output",output)

    action_set = []
    log_prob_set = []

    for i in output:
        m = Categorical(i)
        if(random()<zeta):
            action = m.sample()
        else:
            action = torch.tensor([power_levels-1])
        action_set.append(action)
        log_prob_set.append(m.log_prob(action))


    action = np.array(action_set)+1
    power = np.multiply(action,power_unit)
    log_prob = torch.cat(log_prob_set)

    return power,log_prob


def sumRate(csi,power):
    abs_H = np.reshape(csi, [-1, num_users, num_users])
    abs_H_2 = np.square(abs_H)
    rx_power = np.multiply(abs_H_2, np.reshape(power, [-1, num_users, 1]))
    mask = np.eye(num_users)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + noise_ratio
    rate = np.log(1 + np.divide(valid_rx_power, interference)) / np.log(2.0)
    sum_rate = np.mean(np.sum(rate, axis=1))
    return sum_rate




def calculate_loss(csi,power,log_prob,baseline=0):
    rewards = sumRate(csi,power)


    # Calculate loss
    loss = torch.mul(-log_prob,rewards-baseline)
    loss = loss.sum()
    loss = loss.unsqueeze(0)

    return loss,rewards


def update_parameters(batch_loss,batch_rewards,network,optimizer):
    loss = torch.cat(batch_loss)
    loss = loss.mean()

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    network.loss_history.append(loss.item())
    average_batch_reward = np.mean(batch_rewards)
    network.reward_history.append(average_batch_reward)

    return average_batch_reward


def get_data(h,ind,lim=10000):
    if(ind>=10000):
        ind=0
    return np.array([h[ind]])

def getCSI():
    x = open_dataset("fixedData.txt")
    return x[0]





if __name__ == "__main__":
    #Select GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)



    # Initializing the network and optimizer
    net = PolicyNetwork(num_users, power_levels)
    opt = optim.Adam(net.parameters(), lr=learning_rate)

    #scheduler = optim.lr_scheduler.StepLR(opt,step_size=1000,gamma=0.1)

    best_state = None

    #net.to(device)

    running_reward = 0
    max_reward = -1e20

    x = getCSI()
    index = 0

    for i in range(1,epochs+1):
        batch_loss = []
        batch_rewards = []
        for j in range(batch_size):
            input = get_data(x,index)
            #print(input)
            index+=1
            #print(input)
            power,log_prob = create_power_vector(input,net)
            loss,rewards = calculate_loss(input,power,log_prob,running_reward)
            #print(input)
            #print(loss,rewards)
            batch_loss.append(loss)
            batch_rewards.append(rewards)

        current_reward = update_parameters(batch_loss,batch_rewards,net,opt)
        running_reward = 0.05*current_reward+0.95*running_reward

        if(current_reward>=max_reward):
            max_reward = current_reward
            best_state = net.state_dict()

        #scheduler.step(i)

        if i%100==0:
            print("Epoch: ",i," Current Sum Rate: ",current_reward," Average Sum Rate: ",running_reward, " Maximum Sum Rate: ",max_reward)
            zeta = min(zeta+5e-6,0.97)

    torch.save(net.state_dict(),PATH)
    torch.save(best_state,BESTPATH)


    x = np.array([i for i in range(len(net.reward_history))])
    r = np.array(net.reward_history)
    l = np.array(net.loss_history)

    plt.plot(x, r)
    plt.xlabel("Iterations")
    plt.ylabel("Sum Rate")
    plt.show()

    plt.plot(x, l)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()






