import numpy as np

class MemoryPER:
    def __init__(self, max_length, obs_space, prob_alpha=0.8, prob_beta=0.4):
        self.max_length = max_length
        self.mem_index = 0

        self.data_state = np.zeros((self.max_length, obs_space),dtype=np.float32)
        self.data_action = np.zeros((self.max_length, 1), dtype=np.int64)
        self.data_reward = np.zeros((self.max_length, 1), dtype=np.float32)
        self.data_state_ = np.zeros((self.max_length, obs_space), dtype=np.float32)
        self.data_done = np.zeros((self.max_length, 1), dtype=bool)

        #===== PER
        self.prob_alpha = prob_alpha
        self.prob_beta = prob_beta
        # priorite --> TD error (plus elle est grande, plus on augmente la prob de sampler l'echantillon)
        self.priorities = np.zeros((self.max_length,), dtype=np.float32)

    def remember(self, state, action, reward, state_, done):
        reward = np.expand_dims(reward, axis=-1)
        done = np.expand_dims(done, axis=-1)


        mem_index = self.mem_index % self.max_length
        self.data_state[mem_index] = state.reshape(1, state.shape[0])
        self.data_action[mem_index] = [action]
        self.data_reward[mem_index] = [reward]
        self.data_state_[mem_index] = state_.reshape(1, state_.shape[0])
        self.data_done[mem_index] = [done]
        #===== PER
        #on recup la plus forte priorite vue jusque la; on initialise les val avec elle lors de leur insertion
        max_prio = self.priorities.max() if self.mem_index > 0  else 1.0
        self.priorities[self.mem_index % self.max_length] = max_prio
        #====
        self.mem_index += 1

    def sample(self, batch_size):
        index = min(self.mem_index, self.max_length)
        data_state = self.data_state[:index]
        data_action = self.data_action[:index]
        data_reward = self.data_reward[:index]
        data_state_ = self.data_state_[:index]
        data_done = self.data_done[:index]

        # on veut les priorites connues, donc uniquement la partie remplie de la mem
        if self.mem_index >= self.max_length:
            prios = self.priorities
        else:
            prios = self.priorities[:self.mem_index]

        # on calcule la distrib de prob pour le sampler
        probs = np.power(prios, self.prob_alpha)
        probs /= probs.sum() + 0.001

        index_batch = np.random.choice(np.arange(index), batch_size, replace=False, p=probs)

        state = data_state[index_batch]
        action = data_action[index_batch]
        reward = data_reward[index_batch]
        state_ = data_state_[index_batch]
        done = data_done[index_batch]

        #readjust weights because we sampled in a new prob dist
        total    = min(self.mem_index, self.max_length)
        weights  = (total * probs[index_batch]) ** (-self.prob_beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        return state, action, reward, state_, done, index_batch, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
