
import math
from typing import Dict, List, Optional, Tuple
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

class Actor(nn.Module):

    def __init__(
        self, obs_dim: int, act_dim: int, hidden_units = [32,32]
    ):
    
        super().__init__()
        
        self.hidden1 = nn.Linear(obs_dim,hidden_units[0])
        nn.init.normal_(self.hidden1.weight)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_units[0],hidden_units[1])
        nn.init.normal_(self.hidden2.weight)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_units[1],act_dim)
       
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.hidden1(obs)
        x1 = self.relu1(x)
        x2 = self.hidden2(x1)
        x3 = self.relu2(x2)
        act = self.out(x3)
    
        
        return act
    
    
    #TODO: Collecting data from Issac-Gym
    #Data Required: Action: foot pose-body frame(next step label), Observation: robot_pose, current leg pose. Vx.Vy grid area.