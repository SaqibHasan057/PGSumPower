import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from Datagen import generate_geometry_CSI,generate_rayleigh_CSI
from random import random
from MultiAgent import Network,policyNetworks,create_power_vector,sumRate
from Datagen import open_dataset


#Hyperparameters
learning_rate = 0.01
seed = 5
epsilon = np.finfo(np.float32).eps
epochs = 10000

num_users = 4
power_levels = 5
power_unit = float(1.0/float(power_levels))
noise_ratio = 0.1
batch_size = 32

PATH=" model"

#Setting up the seeds
#torch.manual_seed(seed)
#np.random.seed(seed)
#rng = np.random.RandomState(1)
#zeta = 0



if __name__=="__main__":
    nets,opts = policyNetworks(num_users,power_levels)
    d = torch.load(PATH)

    for i in range(0,len(nets)):
        nets[i].load_state_dict(d[i])

    h,p,r = open_dataset("fixedData.txt")

    ind = 0

    x = h[ind]
    x = np.array([x])
    z0 = p[ind]

    #print(x)

    t=[]

    for i in range(0,1000):
        y = create_power_vector(x,nets)
        t.append(y[0])

    #print(t)
    y = np.sum(t,axis=0)
    y = y/1000

    rate = sumRate(x,y)
    rate2 = sumRate(x,z0)

    print("Calculated:")
    print(y)
    #print(y[1])
    print(rate)
    print()
    print("Actual:")
    print(z0)
    print(rate2)

