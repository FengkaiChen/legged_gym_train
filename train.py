
import math
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from net.net import Actor
import matplotlib.pyplot as plt 


import argparse


#CURRENT TRAINING INFO: OBS - obs_pos: position of 12 joints, obs_vel: velocity of 12 joints
#                       ACT - foot_pos: position(x,y,z) of 4 feet(LF,LH,RF,RH) 

    
def load(dir):
    a = torch.load(dir)
    train_obs = a[:20000,0:24]
    train_act = a[:20000,24:]

    return train_obs,train_act


if __name__ == "__main__":
    
    input_dir = "dataset/train_body.pt"
    batch = 85
    epochs = 300
    learning_rate = 3e-3

    train_obs,train_act = load("dataset/train.pt")
    obs_dim = train_obs.shape[1]
    act_dim = train_act.shape[1]
    train_obs = torch.tensor(train_obs,device='cpu')
    train_act = torch.tensor(train_act,device='cpu')
    train_data = torch.cat((train_obs,train_act), dim = 1)
    trainloader = torch.utils.data.DataLoader(train_data,batch_size = batch,shuffle = True)
    
    
    net = Actor(obs_dim,act_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), learning_rate,weight_decay = 1e-4)
    

    for epoch in range(epochs): 
        running_loss = 0.0
        for i, data in enumerate(trainloader):
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
        
        

PATH = "weight/example.pth"
torch.save(net.state_dict(), PATH)   #save weights

