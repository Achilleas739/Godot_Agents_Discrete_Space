import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from Discrete_models import DiscreteActor, MultiBinaryActor, DiscreteCritic
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch_directml
import numpy as np
from gymnasium import spaces

GREEN = "\033[92m"
RESET = "\033[0m"


def hard_update(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


# -----------------------------
# Subnet for one action type
# -----------------------------

class AgentSubnet:
    def __init__(
        self,
        state_dim,
        action_config_dim,
        hidden_layers,
        lr,
        gamma,
        tau,
        alpha,
        device,
        batch_size,
        max_size,
        target_interval_update,
        name,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_size = max_size
        self.target_interval_update = target_interval_update
        self.action_dim = action_config_dim
        self.multi_binary = False
        self.name = name
        self.reward = 0
        self.evaluate = False
        
        print(f'{GREEN} {self.action_dim} {RESET}')
        
        
        if self.multi_binary:
            self.actor = MultiBinaryActor(state_dim, self.action_dim, hidden_layers).to(
                device
            )
            self.target_entropy = float(self.action_dim) * np.log(2)
            print(f'{GREEN}Actor Multibinary{RESET}')

        else:
            self.actor = DiscreteActor(state_dim, self.action_dim, hidden_layers).to(
                device
            )
            self.target_entropy =  np.log(self.action_dim) * 0.5
        
        # Critics (exclusive)
        self.critic = DiscreteCritic(state_dim, self.action_dim, hidden_layers).to(device)
        self.critic_counter = 0

        self.critic_target = DiscreteCritic(state_dim, self.action_dim, hidden_layers).to(device)

        hard_update(self.critic_target, self.critic)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)
        
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad = True)
        self.alpha = self.log_alpha
        self.alpha_optim = Adam([self.log_alpha], lr=1e-4)
        
        #--------------------------------
        #CHECKS IF DISCRETE ONE ACTION OUTPUT ONLY
        out_action = 1
        if self.multi_binary:
            out_action = self.action_dim
        #------------------------
        
        self.memory = ReplayBuffer(
            max_size = self.max_size,
            input_shape =state_dim,
            n_actions = out_action,
            batch_size = self.batch_size,
        )
    
    
    def _process_action(self, action):
        if self.multi_binary:
            # keep as floats, already [batch_size, action_dim]
            return action.float().view(action.size(0), -1)
        return action


    # -------------------------------------------------
    # Action Sampling
    # -------------------------------------------------
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        logits = self.actor(state)
        
        if self.multi_binary:
            dist = torch.distributions.Bernoulli(logits=logits)
        else:
            dist = torch.distributions.Categorical(logits=logits)

        if self.evaluate:
            action = (dist.probs > 0.5).float() if self.multi_binary else dist.probs.argmax(dim = 1)
            
        else:
            action = dist.sample()
        #print(f'{GREEN} SELECTED {action} {RESET}')
        log_prob = dist.log_prob(action)
        if self.multi_binary:
            log_prob = log_prob.sum(dim=1, keepdim=True) / self.action_dim

        return action.detach().cpu().numpy(), log_prob
    
    def one_hot_action(self, action):
        if not self.multi_binary:
            return F.one_hot(action.long(), num_classes=self.action_dim).float().squeeze(1)
        return action
    # -------------------------------------------------
    # SAC Update
    # -------------------------------------------------
    def update(self):
        if not self.memory.can_sample():
            return None

        # ---------------- Sample batch ----------------
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample_buffer()
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        #action_batch = self.one_hot_action(action_batch)
        
        # ---------------- Critic update ----------------
        with torch.no_grad():
            logits = self.actor(next_state_batch)
            probs = torch.softmax(logits, dim=1)
            log_probs = torch.log(probs + 1e-8)

            q1_all, q2_all = self.critic_target(next_state_batch)
            min_q = torch.min(q1_all, q2_all)

            v_next = (probs * (min_q - self.alpha * log_probs)).sum(dim=1, keepdim = True)

            next_q_value = reward_batch + mask_batch * self.gamma * v_next
        
        
        # Q(s) for all actions
        q1_all, q2_all = self.critic(state_batch)   # [B, action_dim]

        # Pick Q(s, a) using replay actions
        #action_batch = action_batch.long()   # [B,1]

        q1 = q1_all.gather(1, action_batch)   # [B,1]
        q2 = q2_all.gather(1, action_batch)   # [B,1]
        
        
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        critic_loss = q1_loss + q2_loss


        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---------------- Actor update ----------------
        logits = self.actor(state_batch)
        #logits = logits - logits.max(dim=1, keepdim=True)[0]
        probs = torch.softmax(logits/2, dim=1)
        log_probs = torch.log(probs + 1e-8)

        q1_all, q2_all = self.critic(state_batch)
        min_q = (min_q - min_q.mean(dim=1, keepdim=True)) / (min_q.std(dim=1, keepdim=True) + 1e-6)
        
        #print(min_q.mean().item(), min_q.max().item(), min_q.min().item())
        # Discrete SAC objective (Formulation B)
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        
        expected_q = (probs * min_q).sum(dim=1).mean()
        
        # ---------------- Alpha update ----------------
# Weighted alpha loss over action probabilities (discrete)
# Compute alpha loss (expected over all actions)
        probs = torch.softmax(logits, dim=1).detach()  # detach from actor graph
        log_probs_detached = torch.log(probs + 1e-8)   # already detached
        entropy = -(probs * log_probs_detached).sum(dim=1).mean()

        alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().detach()
        alpha_tlogs = self.alpha.detach()

        # ---------------- Soft update of target ----------------
        self.critic_counter += 1
        if self.critic_counter % self.target_interval_update == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # ---------------- Return stats ----------------
        

        #print(probs.mean(dim=0))
        #print("entropy:", entropy.mean().item())
        #print("target:", self.target_entropy)
        #print("alpha:", self.alpha.item())
        #print("alpha * log_probs:", (self.alpha * log_probs).mean().item())
        #print("Q:", min_q.mean().item())

        return (
            q1_loss.item(),
            q2_loss.item(),
            actor_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
            entropy.item(),
            expected_q.item(),
            self.target_entropy,
        )

    def warmup_update(self):
        if not self.memory.can_sample():
            return None

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = (
            self.memory.sample_buffer()
        )

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        action_batch = torch.LongTensor(action_batch).to(self.device)

        # ---------------- Critic update only ----------------
        with torch.no_grad():
            logits_next = self.actor(next_state_batch)
            probs_next = torch.softmax(logits_next, dim=1)
            log_probs_next = torch.log(probs_next + 1e-8)
            q1_next, q2_next = self.critic_target(next_state_batch)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = (probs_next * (min_q_next - self.alpha * log_probs_next)).sum(
                dim=1, keepdim=True
            )
            target_q = reward_batch * 0.1 + mask_batch * self.gamma * v_next

        q1, q2 = self.critic(state_batch)
        q1 = q1.gather(1, action_batch)
        q2 = q2.gather(1, action_batch)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Optionally update target network periodically
        self.critic_counter += 1
        if self.critic_counter % self.target_interval_update == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            q1_loss.item(),
            q2_loss.item(),
            0,
            0,
            0,
            0,
            0,
            self.target_entropy,
        )

    # ---------------- Save / Load ----------------
    def save(self, directory="checkpoints"):
        """
        Save agent subnet state (actor, critic, target, optimizers, log_alpha)
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save actor and actor optimizer
        self.actor.save_checkpoint(os.path.join(directory, f"{self.name}_actor.pt"))

        # Save critic and critic optimizer
        self.critic.save_checkpoint(os.path.join(directory, f"{self.name}_critic.pt"))

        # Save critic target (optional, usually synced during update)
        self.critic_target.save_checkpoint(os.path.join(directory, f"{self.name}_critic_target.pt"))

        torch.save(self.actor_optim.state_dict(), f"{directory}/{self.name}_actor_optim.pt")
        torch.save(self.critic_optim.state_dict(), f"{directory}/{self.name}_critic_optim.pt")

        print(f"AgentSubnet '{self.name}' saved successfully in '{directory}'")

    def load(self, directory="checkpoints"):
        """
        Load agent subnet state from a directory
        """
        #try:
        # Actor
        actor_checkpoint = os.path.join(directory, f"{self.name}_actor.pt")
        
        self.actor.load_checkpoint(actor_checkpoint)
        print(f'{GREEN}ACTOR LOADED{RESET}')
        # Critic
        critic_checkpoint = os.path.join(directory, f"{self.name}_critic.pt")
        
        self.critic.load_checkpoint(critic_checkpoint)
        print(f"{GREEN}CRITIC LOADED{RESET}")
        critic_target_checkpoint = os.path.join(directory, f"{self.name}_critic_target.pt")
        
        # Critic target
        self.critic_target.load_checkpoint(critic_target_checkpoint)
        
        print(f"{GREEN}CRITIC_TARGET LOADED{RESET}")
        
        self.actor_optim.load_state_dict(
            torch.load(f"{directory}/{self.name}_actor_optim.pt")
        )
        print(f'{GREEN}ACTOR OPT LOADED{RESET}')

        self.critic_optim.load_state_dict(
            torch.load(f"{directory}/{self.name}_critic_optim.pt")
        )
        print(f"{GREEN}CRITIC OPT LOADED{RESET}")

        print(f"{GREEN}AgentSubnet '{self.name}' loaded successfully from '{directory}'{RESET}")

        #except FileNotFoundError as e:
        #    print(f"No checkpoint found in '{directory}'. Starting from scratch. ({e})")

    
    def eval(self):
        """
        Set the agent in evaluation mode
        """
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()
        self.evaluate = True
        # log_alpha and optimizers are not affected
        print(f"{GREEN}AgentSubnet '{self.name}' is now in EVAL mode{RESET}")
    
    
    def train(self):
        """
        Set the agent in training mode
        """
        self.actor.train()
        self.critic.train()
        self.critic_target.train()
        print(f"{GREEN}AgentSubnet '{self.name}' is now in TRAIN mode{RESET}")

# -----------------------------
# Hierarchical agent
# -----------------------------
class HierarchicalAgent:
    """
    One agent with multiple subnets + decider network
    """
    def __init__(self, input_dim, action_config, hidden_layers, lr, gamma, tau, alpha, batch_size, max_batch, target_update_interval, policy_name ,device, writer):
        self.device = device
        self.subnets : dict[str, AgentSubnet] = {}
        self.multi_binary_flags = {}
        self.policy_name = policy_name
        self.eval = False
        self.writer = writer
        # Dynamically create all subnets
        print(action_config)
        for key, config in action_config.items():
            subnet_name = policy_name + key
            self.subnets[key] = AgentSubnet(
                state_dim=input_dim,
                action_config_dim = config.n,
                hidden_layers=hidden_layers,
                lr=lr,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                device=self.device,
                batch_size=batch_size,
                max_size=max_batch,
                target_interval_update=target_update_interval,
                name=subnet_name,
            )
            num_of_subnets = len(self.subnets.keys())
            
            
            self.decider = AgentSubnet(
                state_dim=input_dim,
                action_config_dim = num_of_subnets,
                hidden_layers=[256, 256],
                lr=lr,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                device=self.device,
                batch_size=batch_size,
                max_size=max_batch,
                target_interval_update=target_update_interval,
                name=policy_name+'decider',
            )

    def select_action(self, state, evaluate=False):
        # Sample decider first
        decider_action, decider_logp = self.decider.select_action(
            state, evaluate=evaluate
        )
        decider_action = decider_action[0]
        # Choose subnet based on decider
        subnet_keys = list(self.subnets.keys())
        subnet_key = subnet_keys[decider_action]
        
        subnet_action, subnet_logp = self.subnets[subnet_key].select_action(
            state, evaluate=evaluate
        )

        # Return combined action dict
        return (
            subnet_key,
            subnet_action,
            decider_logp + subnet_logp
        )

    def update(self):
        for key,net in {**self.subnets, 'decider':self.decider}.items():
            if net.evaluate:
                continue
            else:
                update = net.update()
                if update is not None:
                    (
                        critic_1_loss,
                        critic_2_loss,
                        policy_loss,
                        alpha_loss,
                        alpha,
                        entropy,
                        expected_q,
                        target_entropy,
                    ) = update

                    critics_counter = net.critic_counter

                    self.writer.add_scalar(
                        f'{key}'+"/loss/alpha_loss", alpha_loss, critics_counter
                    )
                    """
                    agent.writer.add_scalar(
                        f'{key}'+"/loss/prediction", prediction_loss, critics_counter
                    )
                    """
                    
                    self.writer.add_scalar(
                        f"{key}" + "parameters/alpha", alpha, critics_counter
                    )
                    
                    self.writer.add_scalars(
                        f"{key}" + "parameters/entropy",
                        {"entropy": entropy, "target entropy": target_entropy},
                        critics_counter,
                    )

                    self.writer.add_scalar(
                        f'{key}'+"parameters/expected q value", expected_q, critics_counter
                    )

                    self.writer.add_scalars(
                        f"{key}" + "loss/critics",
                        {"critic_1": critic_1_loss, "critic_2": critic_2_loss},
                        critics_counter,
                    )

                    self.writer.add_scalar(
                        f"{key}" + "loss/Actor", policy_loss, critics_counter
                    )

    def memory_update (self, state, action : dict, reward, next_state, masks):
        subnet_keys = list(self.subnets.keys())
        key, action_value = list(action.items())[0]
        decider_action = subnet_keys.index(key)
        #print(f'{RESET}{key}\n\n{decider_action}\n\n{action_value}')
        self.decider.memory.store_transition(state, decider_action, reward, next_state, masks)
        self.subnets[key].memory.store_transition(state, action_value, reward, next_state, masks)
    
    def save(self, directory):
        self.decider.save(os.path.join(directory,"decider"))
        for name,subnet in self.subnets.items():
            if not subnet.evaluate:
                subnet.save(os.path.join(directory,name))
    
    def load(self, directory):
        self.decider.save(os.path.join(directory, "decider"))
        for name, subnet in self.subnets.items():
            subnet.save(os.path.join(directory, name))
    
    def eval(self):
        self.decider.eval()
        for subnet in self.subnets.values():
            subnet.eval()
    
    def train(self):
        self.decider.train()
        for subnet in self.subnets.values():
            if not subnet.evaluate:
                subnet.train()