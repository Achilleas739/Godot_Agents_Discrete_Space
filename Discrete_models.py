import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Bernoulli
import os


epsilon = 1e-6

#initialize policy weights

def weights_init(m):
  if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight, gain = 1)
    torch.nn.init.constant_(m.bias, 0)


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_layers=[128],
        checkpoint_dir="checkpoints",
        name="critic_network",
    ):
        super(DiscreteCritic, self).__init__()

        # Critics Architecture
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        p_layer = num_inputs
        for h in hidden_layers:
            self.layers1.append(nn.Linear(p_layer, h))
            self.layers2.append(nn.Linear(p_layer, h))
            p_layer = h
        self.out_layer1 = nn.Linear(p_layer, num_actions)
        self.out_layer2 = nn.Linear(p_layer, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "sac")

        self.apply(weights_init)

    def forward(self, state):
      
        out1 = state
        # Forward for Critic1
        for layer in self.layers1:
            out1 = F.relu(layer(out1))
        out1 = self.out_layer1(out1)

        out2 = state
        # Forward for Critic2
        for layer in self.layers2:
            out2 = F.relu(layer(out2))
        out2 = self.out_layer2(out2)

        return out1, out2

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))


class Critic(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_layers = [128],
        checkpoint_dir="checkpoints",
        name="critic_network",
    ):
        super(Critic, self).__init__()

      # Critics Architecture
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        p_layer = num_inputs + num_actions
        for h in hidden_layers:
          self.layers1.append(nn.Linear(p_layer, h))
          self.layers2.append(nn.Linear(p_layer, h))
          p_layer = h
        self.out_layer1 = nn.Linear(p_layer, 1)
        self.out_layer2 = nn.Linear(p_layer, 1)


        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "sac")

        self.apply(weights_init)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        out1 = xu
        # Forward for Critic1
        for layer in self.layers1:
          out1 = F.relu(layer(out1))
        out1 = self.out_layer1(out1)
        
        out2 = xu
        # Forward for Critic2
        for layer in self.layers2:
            out2 = F.relu(layer(out2))
        out2 = self.out_layer2(out2)
        
        return out1, out2

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
    
  
  
class DiscreteActor(nn.Module):
  
  def __init__(self, input_dim, output_dim, hidden_layers = [128], checkpoint_dir = 'checkpoints', name = 'actor_network'):
    super(DiscreteActor, self).__init__()
    self.layers = nn.ModuleList()
    previous_dim = input_dim
    for h in hidden_layers:
      self.layers.append(nn.Linear(previous_dim, h))
      previous_dim = h
    self.out = nn.Linear(previous_dim, output_dim)
    
    self.name = name
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'sac')
    
    self.apply(weights_init)
  
  
  def save_checkpoint(self, path):
    torch.save(self.state_dict(), path)
  
  
  def load_checkpoint(self, path):
    self.load_state_dict(torch.load(path))  
  
  
  def forward(self, state):
    for layer in self.layers:
      state = F.relu(layer(state))
    output = self.out(state)
    
    return output
  
  
  def sample(self, state):
    logits = self.forward(state)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob
  
  
  def to(self, device):
    return super(DiscreteActor, self).to(device)
  

class MultiBinaryActor(nn.Module):
  
  def __init__(self, input_dim, output_dim, hidden_layers = [128], action_space = None, checkpoint_dir = 'checkpoints', name = 'actor_network'):
    super(MultiBinaryActor, self).__init__()
    self.layers = nn.ModuleList()
    previous_dim = input_dim
    for h in hidden_layers:
      self.layers.append(nn.Linear(previous_dim, h))
      previous_dim = h
    self.out = nn.Linear(previous_dim, output_dim)
    self.name = name
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'sac')
    
    self.apply(weights_init)
    
    
  def save_checkpoint(self, path):
    torch.save(self.state_dict(), path)
    
    
  def load_checkpoint(self, path):
    self.load_state_dict(torch.load(path))  
  
  
  def forward(self, state):
    for layer in self.layers:
      state = F.relu(layer(state))
    logits = self.out(state)
    probs = torch.sigmoid(logits)
    return logits, probs
  
  
  def sample(self, state):
    logits, _ = self.forward(state)
    dist = Bernoulli(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim = 1, keepdim = True) / logits.shape[1]
    return action, log_prob
  
  
  def to(self, device):
    return super(MultiBinaryActor, self).to(device)



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