import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import PredictiveModel
from Discrete_models import DiscreteActor, MultiBinaryActor, Critic
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch_directml
import numpy as np
from ActorCritic_subnets import AgentSubnet, HierarchicalAgent

GREEN = "\033[92m"
RESET = "\033[0m"


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)






class MultiAgentSAC:

    def __init__(
                self,
                num_inputs,
                action_space,
                gamma,
                tau,
                alpha,
                hidden_size,
                sac_lr,
                icm_lr,
                agent_lr,
                target_update_interval,
                exploration_scaling_factor,
                policy_name,
                log_root = 'runs',
                test = False,
                batch_size = 128,
                max_batch = int(1e6),
                hierarchical : bool = False
                ):
    
        self.hierarchical = hierarchical
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.exploration_scaling_factor = exploration_scaling_factor
        self.policy_name = policy_name
        self.evaluate = test
        self.test = test
        self.rewards = dict()
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            try:
                self.device = torch_directml.device()
            except Exception:
                self.device = torch.device('cpu')
        
        print(f"{GREEN}Using device: {self.device}{RESET}")
        
        # Logging directories
        if test is False:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.log_dir = os.path.join(log_root, policy_name + timestamp)
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_dir))
        
        action_conf = action_space["movement"]
        
        

        if hierarchical:
            self.agent_policy = HierarchicalAgent(
                input_dim=num_inputs,
                action_config=action_space,
                hidden_layers=[256, 256],
                lr=agent_lr,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                device=self.device,
                batch_size=batch_size,
                max_batch=max_batch,
                target_update_interval=target_update_interval,
                policy_name=policy_name,
                writer = self.writer
            )
        
        else:
            self.agent_policy = AgentSubnet(
                state_dim = num_inputs,
                action_config_dim = action_conf.n,
                hidden_layers = [256, 256],
                lr = agent_lr,
                gamma = gamma,
                tau = tau,
                alpha = alpha,
                device = self.device,
                batch_size = batch_size,
                max_size = max_batch,
                target_interval_update = target_update_interval,
                name = policy_name,
                )
            
        
            
        
    
    
    def select_action(self, state):
        return self.agent_policy.select_action(state, self.evaluate)
    
    
    def update_parameters(self):
        if self.hierarchical:
            self.agent_policy.update()
            return None
        elif self.agent_policy.memory.can_sample():
            return self.agent_policy.update()
        return None
    
    def warmup_update_parameters(self):
        if self.hierarchical:
            return None
        elif self.agent_policy.memory.can_sample():
            return self.agent_policy.warmup_update()
        return None
    
    def save(self, directory='checkpoints'):
        directory = os.path.join(directory, f"{self.policy_name}")
        os.makedirs(directory, exist_ok=True)
        self.agent_policy.save(directory = directory)
    
    
    def load(self, directory="checkpoints", evaluate = False):
        directory = os.path.join(directory, f"{self.policy_name}")
        
        try:
            self.agent_policy.load(directory)
            print(f"{GREEN} ALL CHECKPOINTS LOADED SUCCESSFULLY{RESET}")
        
        except:  
            if evaluate:
                pass#raise Exception("Unable to evaluate models without a loaded checkpoint")
            else:
                print("Unable to load models. Starting from scratch")
            
        if evaluate:
            self.evaluate = True
            self.agent_policy.eval()
            print(f"{GREEN}EVAL MODE{RESET}")

        else:
            self.agent_policy.train()
            print(f"{GREEN}TRAIN MODE{RESET}")

    def get_rewards(self):
        return self.rewards
    
    def reset_rewards(self):
        for id in self.rewards.keys():
            self.rewards[id] = 0