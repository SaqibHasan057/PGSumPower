import numpy as np
from Datagen import generate_geometry_CSI,generate_rayleigh_CSI



def sumRate(num_users,csi,power,noise_ratio):
    abs_H = np.reshape(csi, [-1, num_users, num_users])
    abs_H_2 = np.square(abs_H)
    rx_power = np.multiply(abs_H_2, np.reshape(power, [-1, num_users, 1]))
    mask = np.eye(num_users)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + noise_ratio
    rate = np.log(1 + np.divide(valid_rx_power, interference)) / np.log(2.0)
    sum_rate = np.mean(np.sum(rate, axis=1))
    return sum_rate



def getSolution(csi):
    maxSumRate = -np.inf
    maxPowRate = []
    for i in range(1,6):
        for j in range(1,6):
            for k in range(1,6):
                for l in range(1,6):
                    pow = np.array([i,j,k,l])
                    pow = pow/5
                    x = sumRate(4,csi,pow,0.1)
                    if(x>maxSumRate):
                        maxSumRate = x
                        maxPowRate = pow

    return maxSumRate,maxPowRate


def storeInFile(x,pow,sumRate,filename):
    f = open(filename,"w")

    for i in range(0,len(x)):
        x_temp = x[i]
        pow_temp = pow[i]
        sumRate_temp = sumRate[i]

        s = ""
        for j in x_temp:
            s+=str(j)+";"

        for j in pow_temp:
            s+=str(j)+";"

        s+=str(sumRate_temp)+"\n"

        f.write(s)

    f.close()


x = generate_geometry_CSI(4,10000,np.random.RandomState(1))
pow_y = []
sum_y = []
for i in x:
    a,b = getSolution(i)
    pow_y.append(b)
    sum_y.append(a)


storeInFile(x,pow_y,sum_y,"fixedData.txt")

