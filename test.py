import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from net.net import Actor


#CURRENT testING INFO: obs: obs

    
def load():
    a = torch.load("dataset/train.pt")
    test_obs = a[10000:13000,0:24]
    test_act = a[10000:13000,24:]
    
    return test_obs,test_act



if __name__ == "__main__":
    

    batch = 8
    epochs = 300
    learning_rate = 3e-4

    test_obs,test_act = load()
    obs_dim = test_obs.shape[1]
    act_dim = test_act.shape[1]
    # test_obs = torch.tensor(test_obs,devices='cpu')
    # test_act = torch.tensor(test_act,device='cpu')

    test_data = torch.cat((test_obs,test_act), dim = 1)
    test_data = test_data.cpu()
    testloader = torch.utils.data.DataLoader(test_data,batch_size = batch,shuffle = True)


    net = Actor(obs_dim,act_dim)    #load weights

    PATH = "weight/example.pth"
    net.load_state_dict(torch.load(PATH))
    net.eval()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), learning_rate,weight_decay = 1e-4)
    

    for epoch in range(epochs): 
        running_loss = 0.0
        for i, data in enumerate(testloader):
            obs, act = data[:,:24],data[:,24:]
            optimizer.zero_grad()
            act_pred = net(obs)
            loss = criterion(act_pred, act)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if i % 200 == 199: 
            #     print(f'epoch: {epoch + 1}, steps: {i + 1:5d} loss: {loss:.3f}')
            #     running_loss = 0.0
        print(f'epoch: {epoch + 1},  loss: {loss :.3f}')
        
    