import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from Datagen import generate_geometry_CSI,generate_rayleigh_CSI,open_dataset
from random import random
from sklearn.preprocessing import normalize


#Hyperparameters
learning_rate = 0.001
seed = 5
epsilon = np.finfo(np.float32).eps
epochs = 9000

num_users = 4
power_levels = 5
power_unit = float(1.0/float(power_levels))
noise_ratio = 0.1
batch_size = 1
dataset_size = 500

PATH=" model"

#Setting up the seeds
#torch.manual_seed(seed)
#np.random.seed(seed)
#rng = np.random.RandomState(1)
#zeta = 0.99
zeta = 1

class Network(nn.Module):


    def __init__(self,num_users,power_levels):
        super(Network,self).__init__()

        self.fc1 = nn.Linear(num_users*num_users, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64,power_levels)


    def forward(self,x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = torch.softmax(self.output(x),dim=-1)
        return x

    """
    def __init__(self, num_users, power_levels):
        super(Network, self).__init__()


        self.lstm = nn.LSTM(num_users*num_users,128)
        self.output = nn.Linear(128, power_levels)

    def forward(self, x):
        x = x.view(1,1,16)
        x = self.lstm(x)
        x = torch.softmax(self.output(x[0]), dim=-1)
        return x
    """


def policyNetworks(num_users,power_levels):
    nets = []
    opts = []

    for i in range(num_users):
        t1 = Network(num_users,power_levels)
        t2 = optim.Adam(t1.parameters(), lr=learning_rate)
        nets.append(t1)
        opts.append(t2)

    return nets,opts



def create_power_vector(x,nets,t=False):
    input = torch.from_numpy(x).float()
    outs = []
    for i in nets:
        outs.append(i(input))

    #print("Input:",input)
    if(t):
        print("Output:",outs)

    action_set = []
    log_prob_set = []

    ind = 1
    for i in outs:
        #print(ind,i)
        m = Categorical(i)
        if (random() < zeta):
            a = m.sample()
        else:
            a = torch.tensor([np.random.randint(num_users)])

        action_set.append(a)
        log_prob_set.append(m.log_prob(a))
        ind+=1
    #print(log_prob_set)
    #exit()

    action = np.array(action_set)+1
    power = np.multiply(action,power_unit)

    #print(action,power)
    #print()

    return power,log_prob_set


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
    #print(rewards)

    loss = []
    for i in log_prob:
        loss.append(-i*(rewards-1))

    loss = torch.cat(loss)

    return loss,rewards


def update_parameters(batch_loss,batch_rewards,nets,opts):
    #print(batch_loss)
    loss = torch.stack(batch_loss)
    #print(loss)
    loss = loss.mean(dim=0)
    #print(loss)

    #exit()



    for i in range(0,len(opts)):
        opts[i].zero_grad()
        loss[i].backward(retain_graph=True)
        #print(nets[i].output.weight.grad)
        #nn.utils.clip_grad_norm(nets[i].parameters(),40)
        opts[i].step()

    average_batch_reward = np.mean(batch_rewards)
    return average_batch_reward


def get_data(h,ind,lim=10000):
    return np.array([h[ind]])

def getCSI():
    x = open_dataset("fixedData.txt")
    return x[0]





if __name__ == "__main__":


    #Initializing the network and optimizer
    nets,opts = policyNetworks(num_users,power_levels)

    #Getting the data from dataset
    x = getCSI()
    index = 0

    #Setting up the variables
    best_state = None
    running_reward = 0
    max_reward = -1e20

    print("Starting!")
    for i in range(1,epochs+1):
        batch_loss = []
        batch_rewards = []
        for j in range(batch_size):
            input = get_data(x,index)

            #input = normalize(input)

            #print(input)
            index = (index+1)%dataset_size
            #print(input)
            power,log_prob = create_power_vector(input,nets)
            loss,rewards = calculate_loss(input,power,log_prob,running_reward)
            #print(input)
            #print(loss,rewards)
            batch_loss.append(loss)
            batch_rewards.append(rewards)

        current_reward = update_parameters(batch_loss,batch_rewards,nets,opts)
        running_reward = 0.01*current_reward+0.99*running_reward

        max_reward = max(max_reward,current_reward)

        #scheduler.step(i)

        if i%1000==0:
            print("Epoch: ",i," Current Sum Rate: ",current_reward," Average Sum Rate: ",running_reward, " Maximum Sum Rate: ",max_reward)

    d = {}
    for i in range(0,len(nets)):
        d[i] = nets[i].state_dict()

    torch.save(d,PATH)



    """
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
    """





