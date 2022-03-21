import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


class AgentDDQN:
    def __init__(self, action_value, target, memory, control, criterion, optimizer, scheduler, action_space, batch_size, gamma, l2_lambda, freq_update_target, is_regul, is_per, device):
        self.action_value = action_value    # estimateur de Q value
        self.target = target
        self.memory = memory                # memoire des coups joues, pour le training
        self.control = control              # selecteur de l'action a joue selon la politique
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        ######
        self.action_space = action_space
        self.BATCH_SIZE = batch_size
        #####
        self.gamma = gamma
        self.l2_lambda = l2_lambda
        #####
        self.is_per = is_per
        self.is_regul = is_regul
        self.device = device

        self.offset_estim_target = 0
        self.freq_update_target = freq_update_target

    #return action indice in the action_space
    def act(self, observation, eps, training=True):
        with torch.no_grad():
            # on evalue Q(s,a) pour tout a
            observation = torch.tensor(observation, dtype=torch.float32)
            Q = self.action_value(observation.to(self.device))
            # on passe au control
        return self.control(Q, eps, training)

    def remember(self, observation, action, reward, next_observation, done):
        #on passe a la memoire
        self.memory.remember(observation, action, reward, next_observation, done)
        return

    def update_target(self):
        self.target.load_state_dict(self.action_value.state_dict())

    def learn(self):
        # load a batch and train
        # on ne commence a apprendre qu'apres avoir suffisamment d'elements dans la memoire pour remplir au moins un batch
        if self.memory.mem_index < self.BATCH_SIZE:
            return None

        self.optimizer.zero_grad()

        # charge un batch (X, a, r, X_, done) et on convertit en tensors
        if self.is_per:
            observations, actions, rewards, observations_, dones, indices, weights = self.memory.sample(self.BATCH_SIZE)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            observations, actions, rewards, observations_, dones = self.memory.sample(self.BATCH_SIZE)

        observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        observations_ = torch.tensor(observations_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int64).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)

        # estim(X,a) --> y^
        cur_Q = self.action_value(observations)

        # je ne veux que les val des actions faites et enregistrees dans la mem (1 seule col)
        cur_Q = torch.gather(cur_Q, 1, actions).squeeze(1)

        # on evalue la target
        next_Q = self.target(observations_).to(self.device)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q * (1 - dones.squeeze(1))

        ###################
        if self.is_per:
            # loss & back prop
            loss = (cur_Q - expected_Q).pow(2) * weights.to(self.device)
            prios = loss + 1E-5
            loss = loss.mean().to(self.device)
            self.memory.update_priorities(indices, prios.data.cpu().numpy())
        else :
            loss = self.criterion(cur_Q, expected_Q)

        if self.is_regul:
            # L2 regularization
            l2_norm = 0
            for p in self.action_value.parameters():
                l2_norm += p.pow(2.0).sum()
            loss += self.l2_lambda * l2_norm

        loss.backward()
        self.optimizer.step()

        log_loss = loss.cpu().detach().numpy()

        """
        self.offset_estim_target += 1
        if self.offset_estim_target >= self.freq_update_target:
            self.update_target()
            self.offset_estim_target = 0
        """

        return log_loss

    def update_target(self):
        self.target.load_state_dict(self.action_value.state_dict())