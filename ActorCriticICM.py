import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import Actor, Critic, PredictiveModel
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch_directml


GREEN = "\033[92m"
RESET = "\033[0m"


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class AgentPolicy:

    def __init__(self, name, num_inputs, action_space, hidden_size, lr, device):
        self.name = name
        self.device = device
        self.reward = 0
        self.memory = None
        self.policy = Actor(
            num_inputs = num_inputs,
            num_actions = action_space.shape[0],
            hidden_dim = hidden_size,
            action_space = action_space,
            name = "actor_network_" + str(name),
        ).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        return action.detach().cpu().numpy()[0]




class MultiAgentSAC:

    def __init__(
                self,
                agent_names,
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
                test = False
                ):
    
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.exploration_scaling_factor = exploration_scaling_factor
        self.policy_name = policy_name
        self.evaluate = test
        self.critics_counter = 0
        self.test = test
        
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

        action_dim = action_space.shape[0]
        
        #Initialize Critic model
        self.critic = Critic(num_inputs, action_dim, hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=sac_lr)

        self.critic_target = Critic(num_inputs, action_dim, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Initialize Predictive model
        self.predictive_model = PredictiveModel(num_inputs, action_dim, hidden_size).to(self.device)
        self.predictive_model_optim = Adam(self.predictive_model.parameters(), lr=icm_lr)
        
        #Initialize Actor models
        self.policies = {}
        
        for name in agent_names:
            agent_policy = AgentPolicy(name, num_inputs, action_space, hidden_size, agent_lr, self.device)
            self.policies[name] = agent_policy
        
        print(
            f"{GREEN}PredictiveModel on: {next(self.predictive_model.parameters()).device}"
            f" | Critic on: {next(self.critic.parameters()).device}{RESET}"
        )
    
    def select_action(self, agent_name, state):
        return self.policies[agent_name].select_action(state, self.evaluate)


    def update_parameters(self, memory: ReplayBuffer, agent_name):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = (memory.sample_buffer())

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        predicted_next_state = self.predictive_model(state_batch, action_batch)

        # Calculate prediction loss
        prediction_error = F.mse_loss(predicted_next_state, next_state_batch)
        prediction_error_no_reduction = F.mse_loss(predicted_next_state, next_state_batch, reduce = False)
        
        scaled_intrinsic_reward = prediction_error_no_reduction.mean(dim = 1)
        scaled_intrinsic_reward = self.exploration_scaling_factor * torch.reshape(scaled_intrinsic_reward, (memory.batch_size, 1))
        
        reward_batch = reward_batch + scaled_intrinsic_reward
        
        
        with torch.no_grad():
            
            next_state_action, next_state_log_pi, _ = self.policies[agent_name].policy.sample( next_state_batch )
            
            qf1_next_target, qf2_next_target = self.critic_target( next_state_batch, next_state_action )
            
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        # Check device of parameters before optimizer step

        # Update Critic network
        self.critic_optim.zero_grad()
        qf_loss.backward()

        self.critic_optim.step()

        # Update the predictive network
        self.predictive_model_optim.zero_grad()
        prediction_error.backward()
        
        self.predictive_model_optim.step()

        pi, log_pi, _ = self.policies[agent_name].policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update the policy network
        self.policies[agent_name].optimizer.zero_grad()
        policy_loss.backward()
        
        self.policies[agent_name].optimizer.step()


        alpha_loss = torch.tensor(0.0).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha)


        self.critics_counter += 1

        if self.critics_counter % self.target_update_interval == 0 and self.critics_counter > 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            prediction_error.item(),
            alpha_tlogs.item(),
        )

    def save(self, directory='checkpoints'):
        directory = os.path.join(directory, f"{self.policy_name}")
        os.makedirs(directory, exist_ok=True)
        self.critic.save_checkpoint(os.path.join(directory, 'critic.pt'))
        #self.critic_target.save_checkpoint(os.path.join(directory, '_critic_target'))
        self.predictive_model.save_checkpoint(os.path.join(directory, 'icm.pt'))
        for name, policy in self.policies.items():
            policy.policy.save_checkpoint(os.path.join(directory, f'policy_{name}.pt'))

    def load(self, directory="checkpoints", evaluate = False):
        directory = os.path.join(directory, f"{self.policy_name}")
        try:
            self.critic.load_checkpoint(os.path.join(directory, "critic.pt"))
            self.critic_target.load_checkpoint(os.path.join(directory, "critic.pt"))
            self.predictive_model.load_checkpoint(os.path.join(directory, "icm.pt"))
            for name, policy in self.policies.items():
                policy.policy.load_checkpoint(os.path.join(directory, f"policy_{name}.pt"))
        except:  
            if evaluate:
                raise Exception("Unable to evaluate models without a loaded checkpoint")
            else:
                print("Unable to load models. Starting from scratch")
        
        if self.evaluate:
            print(f'{GREEN}EVAL MODE{RESET}')
            self.predictive_model.eval()
            self.critic.eval()
            self.critic_target.eval()
            for policy in self.policies.values():
                policy.policy.eval()
        
        else:
            self.predictive_model.train()
            self.critic.train()
            self.critic_target.train()
            for name, policy in self.policies.items():
                policy.policy.train()
        print(f"{GREEN} ALL CHECKPOINTS LOADED SUCCESSFULLY{RESET}")
    
    def get_rewards(self):
        
        rewards = dict()
        
        for id in self.policies.keys():
            rewards[str(id)] = self.policies[id].reward
        
        return rewards
    
    def reset_rewards(self):
        for id in self.policies.keys():
            self.policies[id].reward = 0