
import math
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 


    
def load(dir):
    a = torch.load(dir)
    train_act=[]
    train_obs=[]
    for i in range(20000):
        if(i % 50 == 0):
            train_obs.append(np.array(a[i,0:24].cpu())) 
            train_act.append(np.array(a[i,24:].cpu()))

    return np.array(train_obs),np.array(train_act)


if __name__ == "__main__":
    obs,act = load("dataset/base.pt")
    # print(obs.shape)
    obs_pos = obs[:,:12]
    obs_vel = obs[:,12:]
    base = act[:,:3]
    lf = act[:,3:6]
    lh = act[:,6:9]
    rf = act[:,9:12]
    rh =  act[:,12:15]




    print(lf - base)
    fig, ax = plt.subplots(3, 3)
    fig.suptitle('Left foreleg' )
    ax[0,0].plot(lf[:,0])
    ax[0,0].set_title("lf_x")
    ax[0,1].plot(lf[:,1])
    ax[0,1].set_title("lf_y")
    ax[0,2].plot(lf[:,2])
    ax[0,2].set_title("lf_z")

    ax[1,0].plot(base[:,0])
    ax[1,0].set_title("base_x")
    ax[1,1].plot(base[:,1])
    ax[1,1].set_title("base_y")
    ax[1,2].plot(base[:,2])
    ax[1,2].set_title("base_z")


    ax[2,0].plot(lf[:,0] - base[:,0])
    ax[2,0].set_title("lf_x wrt base")
    ax[2,1].plot(lf[:,1] - base[:,1])
    ax[2,1].set_title("lf_y wrt base")
    ax[2,2].plot(lf[:,1] - base[:,2])
    ax[2,2].set_title("lf_zwrt base")

    
    plt.legend()
    plt.show()
