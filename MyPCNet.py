import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Datagen import open_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Datagen import generate_geometry_CSI,generate_rayleigh_CSI


class network(nn.Module):

    def __init__(self,user_num,noise_ratio):
        super(network,self).__init__()

        self.user_num = user_num
        self.noise_ratio = noise_ratio

        self.fc1 = nn.Linear(self.user_num*self.user_num,64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,32)
        self.bn2 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32,4)


    def forward(self, input):
        x = input.view(-1,self.user_num*self.user_num)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.sigmoid(self.out(x))

        return x

    def sumRate(self,csi,power,device="cpu"):
        abs_H = csi.view(-1,self.user_num,self.user_num)
        abs_H_2 = torch.mul(abs_H,abs_H)
        rx_power = torch.mul(abs_H_2,power.view(-1,self.user_num,1))
        mask = torch.eye(self.user_num).to(device)
        valid_rx_power = torch.sum(torch.mul(rx_power,mask),dim=1)
        interference = torch.sum(torch.mul(rx_power,1-mask),dim=1)+self.noise_ratio
        rate = torch.log2(1+torch.div(valid_rx_power,interference))
        sum_rate = torch.mean(torch.sum(rate,dim=1))
        return sum_rate


def initialize(user_num,noise_ratio,learning_rate,device):
    neural_net = network(user_num,noise_ratio)
    neural_net.to(device)
    optimizer = torch.optim.Adam(neural_net.parameters(),lr=learning_rate)

    return neural_net,optimizer


def train_network(batch,neural_net,optimizer,device):
    x = torch.from_numpy(batch).float().to(device)
    out = neural_net(x)

    batchSumRate = neural_net.sumRate(x,out,device)
    loss = -batchSumRate

    #print(x[0])
    #print(loss)

    #Gradient Descent Step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return batchSumRate


def test_network(test_set,neural_net,device):
    x = torch.from_numpy(test_set).float().to(device)
    out = neural_net(x)

    testSumRate = neural_net.sumRate(x,out,device)

    return testSumRate

def getCSI():
    x = open_dataset("fixedData.txt")
    return x[0]

def next_batch(dataset,index,batch_size,dataset_size):

    if(index+batch_size>=dataset_size):
        return dataset[index:]
    else:
        return dataset[index:index+batch_size]


def plotGraph(x,y,x_label,y_label,title):
    plt.plot(x,y)
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.title = title
    plt.grid()

    plt.show()




user_num = 4
noise_ratio = 0.1
learning_rate = 0.001
batch_size = 256
epochs = 10000
PATH = "customPCNET"

loss = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__=="__main__":

    #Initialize neural net
    neural_net,optimizer = initialize(user_num,noise_ratio,learning_rate,device)


    #Initialize dataset
    dataset = getCSI()
    train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=2)
    train_set_size = len(train_set)
    ind = 0




    #Model Save Variables
    best_parameters = None
    maximum_sumRate = -1e20

    print("Training started!!")


    #Train
    for i in range(epochs):

        ##Get batch
        """
        batch = next_batch(train_set,ind,batch_size,train_set_size)
        ind+=batch_size
        if(ind>=train_set_size):
            ind=0
        """

        batch = generate_geometry_CSI(user_num,batch_size,np.random.RandomState())
        #print(batch[0])



        ##Train and update network on batch
        batchSumRate = train_network(batch,neural_net,optimizer,device)
        loss.append(batchSumRate)


        ##Test and store network

        if(i%100==0):
            #testSumRate = test_network(test_set, neural_net,device)
            testSumRate = test_network(generate_geometry_CSI(user_num,1000,np.random.RandomState(23)),neural_net,device)

            if(testSumRate>maximum_sumRate):
                maximum_sumRate=testSumRate
                best_parameters = neural_net.state_dict()

            print("Iteration:",i," Iteration Sum Rate:",batchSumRate.item()," Test Sum Rate:",testSumRate.item()," Maximum Sum Rate",maximum_sumRate.item())


    print("Training finished!")

    testSumRate = test_network(test_set, neural_net, device)

    print("Final Test Sum Rate:",testSumRate.item()," Maximum Sum Rate",maximum_sumRate.item())

    torch.save(best_parameters,PATH)

    print("Plotting Graph!!")
    x = [i for i in range(1,epochs+1)]
    y = loss
    plotGraph(x,y,"Iterations","Sum rate","Batch sum rate against iterations")

    print("Code run complete!!")



















