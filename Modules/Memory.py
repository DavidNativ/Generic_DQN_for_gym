import numpy as np

class Memory:
    def __init__(self, max_length, obs_space):
        self.max_length = max_length
        self.mem_index = 0

        self.data_state = np.zeros((self.max_length, obs_space),dtype=np.float32)
        self.data_action = np.zeros((self.max_length, 1), dtype=np.int64)
        self.data_reward = np.zeros((self.max_length, 1), dtype=np.float32)
        self.data_state_ = np.zeros((self.max_length, obs_space), dtype=np.float32)
        self.data_done = np.zeros((self.max_length, 1), dtype=bool)

    def remember(self, state, action, reward, state_, done):
        #rewards = rewards.reshape((-1, 1))
        #dones = dones.reshape((-1, 1))
        reward = np.expand_dims(reward, axis=-1)
        done = np.expand_dims(done, axis=-1)


        mem_index = self.mem_index % self.max_length
        self.data_state[mem_index] = state.reshape(1, state.shape[0])
        self.data_action[mem_index] = [action]
        self.data_reward[mem_index] = [reward]
        self.data_state_[mem_index] = state_.reshape(1, state_.shape[0])
        self.data_done[mem_index] = [done]
        self.mem_index += 1

    def sample(self, batch_size):
        index = min(self.mem_index, self.max_length)

        data_state = self.data_state[:index]
        data_action = self.data_action[:index]
        data_reward = self.data_reward[:index]
        data_state_ = self.data_state_[:index]
        data_done = self.data_done[:index]

        index_batch = np.random.choice(np.arange(index), batch_size, replace=False)

        state = data_state[index_batch]
        action = data_action[index_batch]
        reward = data_reward[index_batch]
        state_ = data_state_[index_batch]
        done = data_done[index_batch]

        return state, action, reward, state_, done
