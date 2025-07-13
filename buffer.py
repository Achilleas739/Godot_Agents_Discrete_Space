import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, batch_size):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros(
            (self.mem_size, input_shape), dtype=np.float32
        )
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.batch_size = batch_size

    def can_sample(self):
        if self.mem_ctr > (self.batch_size) :
            return True
        return False

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state.astype(np.float32)
        self.next_state_memory[index] = next_state.astype(np.float32)
        self.action_memory[index] = action.astype(np.float32)
        self.reward_memory[index] = reward.astype(np.float32)
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones