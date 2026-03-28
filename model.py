import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

#initialize policy weights

def weights_init(m):
  if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight, gain = 1)
    torch.nn.init.constant_(m.bias, 0)
    
    
class Critic(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        checkpoint_dir="checkpoints",
        name="critic_network",
    ):
        super(Critic, self).__init__()

        # Critic1 Architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # Critic2 Architecture
        self.linear5 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "sac")

        self.apply(weights_init)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        # Forward for Critic1
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = self.linear4(x1)
        # Forward for Critic2
        x2 = F.relu(self.linear5(xu))
        x2 = F.relu(self.linear6(x2))
        x2 = F.relu(self.linear7(x2))
        x2 = self.linear8(x2)

        return x1, x2

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
    
    
class Actor(nn.Module):
  
  def __init__(self, num_inputs, num_actions, hidden_dim, action_space = None, checkpoint_dir = 'checkpoints', name = 'actor_network'):
    super(Actor, self).__init__()
    
    self.linear1 = nn.Linear(num_inputs, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = nn.Linear(hidden_dim, hidden_dim)
    self.out = nn.Linear(hidden_dim, num_actions)
    
    self.log_std_linear = nn.Linear(hidden_dim, num_actions)
    
    self.name = name
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'sac')
    
    self.apply(weights_init)
    
      
  def save_checkpoint(self, path):
    torch.save(self.state_dict(), path)
    
    
  def load_checkpoint(self, path):
    self.load_state_dict(torch.load(path))  
  
  
  def forward(self, state):
    x = F.relu(self.linear1(state))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    out = self.out(x)
    
    return out
  
  
  def sample(self, state):
    out = self.forward(state)
    dist = Categorical(logits=out)
    action = dist.sample()
    return action
  
  
  def to(self, device):
    return super(Actor, self).to(device)
  
  
class PredictiveModel(nn.Module):
  
  def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir = 'checkpoints', name = 'predictive_network'):
    super(PredictiveModel, self).__init__()
    
    self.fc1 = nn.Linear(num_inputs + num_actions, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, num_inputs)
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
    
    self.apply(weights_init)
  
  def forward(self,state, action):
    x = torch.cat([state, action], dim = 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    predicted_state = self.fc3(x)
    return predicted_state
  
  def save_checkpoint(self,path):
    torch.save(self.state_dict(), path)
    
    
  def load_checkpoint(self, path):
    self.load_state_dict(torch.load(path)) 