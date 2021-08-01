import numpy as np

def generate_geometry_CSI(user_num, batch_size, rng):  # generate a batch of structural data in RoundRobin paper.
    area_length = 10
    alpha = 2
    tx_pos = np.zeros([batch_size, user_num, 2])
    rx_pos = np.zeros([batch_size, user_num, 2])
    rayleigh_coeff = np.zeros([batch_size, user_num, user_num])
    for i in range(batch_size):
        tx_pos[i, :, :] = rng.rand(user_num, 2) * area_length
        rx_pos[i, :, :] = rng.rand(user_num, 2) * area_length
        rayleigh_coeff[i, :, :] = (np.square(rng.randn(user_num, user_num)) + np.square(rng.randn(user_num, user_num))) / 2

    tx_pos_x = np.reshape(tx_pos[:, :, 0], [batch_size, user_num, 1]) + np.zeros([1, 1, user_num])
    tx_pos_y = np.reshape(tx_pos[:, :, 1], [batch_size, user_num, 1]) + np.zeros([1, 1, user_num])
    rx_pos_x = np.reshape(rx_pos[:, :, 0], [batch_size, 1, user_num]) + np.zeros([1, user_num, 1])
    rx_pos_y = np.reshape(rx_pos[:, :, 1], [batch_size, 1, user_num]) + np.zeros([1, user_num, 1])
    d = np.sqrt(np.square(tx_pos_x - rx_pos_x) + np.square(tx_pos_y - rx_pos_y))
    G = np.divide(1, 1 + d**alpha)
    G = G * rayleigh_coeff
    return np.sqrt(np.reshape(G, [batch_size, user_num ** 2]))


def generate_rayleigh_CSI(K, num_H, rng):
    X = np.zeros((num_H, K ** 2))
    for loop in range(num_H):
        #  generate num_H samples of CSI one by one instead of generating all together in order to generate the same samples with different num_H and the same rng.
        CH = 1 / np.sqrt(2) * (rng.randn(1, K ** 2) + 1j * rng.randn(1, K ** 2))
        X[loop, :] = abs(CH)
    return X



def open_dataset(filename):
    f = open(filename,"r")

    h = []
    p = []
    r = []

    for i in f.readlines():
        x = i.split(sep=";")
        z=[]
        for j in x:
            z.append(float(j))

        xs = z[:16]
        #xs = [round(elem,2) for elem in xs]
        h.append(xs)
        p.append(z[16:20])
        r.append(z[20])



    h = np.array(h)
    p = np.array(p)
    r = np.array([r])

    #print(h.shape)
    #print(h[0])
    #print(p.shape)
    #print(p[0])
    #print(r.shape)
    #print(r[0][0])

    return h,p,r



if __name__=="__main__":
    #x = generate_geometry_CSI(2,1,np.random.RandomState(1))
    #print(x)

    h,p,r = open_dataset("fixedData.txt")

    print(np.mean(r[0]))