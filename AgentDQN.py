import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import gym

from Modules.Memory import Memory
from Modules.MemoryPER import MemoryPER
from Modules.Control import Control
from Modules.SimpleNN import SimpleNN
from Modules.Trainer import Trainer





#=============================================#
# GENERAL REINFORCEMENTDeep Q Learning AGENT  #
#=============================================#

#main class: the agent acts, remembers and learns
class AgentDQN:
    def __init__(self, action_value, memory, control, criterion, optimizer, scheduler, action_space, batch_size, gamma, l2_lambda, is_regul, is_per, device):
        self.action_value = action_value    # estimateur de Q value
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
        self.is_regul =is_regul
        self.device = device

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
        next_Q = self.action_value(observations_).to(self.device)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q * (1-dones.squeeze(1))

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
        return log_loss
# -----------------------------------------------


#this is the function that main is calling for instanciate and train the model
def DQL(name, LR, BATCH_SIZE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN, L2_LAMBDA,
        MEM_SIZE, hidden_size,
        nb_epochs, log_interval,
        step_valid, nb_valid, render_valid,
        nb_test, render_test,
        is_PER=False, is_regul=False
        ):

    print( "\n#=========================#==========================#\n"
           "# GENERAL REINFORCEMENT Simple Deep Q Learning AGENT #\n"
           "#====================================================#\n")
    print(f"Model: {name}\n")



    #env = gym.make("MountainCar-v0")
    #env = gym.make("Acrobot-v1")

    """
    env = gym.make("CartPole-v0")
    # 1000 epochs
    LR = 1E-3
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON = 1
    EPSILON_DECAY = 2E-4
    EPSILON_MIN = 0.05
    L2_LAMBDA = 1E-5
    """
    env = gym.make("LunarLander-v2")
    ##

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # NN
    action_value = SimpleNN(input_size=env.observation_space.shape[0],
                             hidden_size_in=hidden_size, hidden_size_out=hidden_size,
                             output_size=env.action_space.n).to(device)

    # Replay Buffer
    if not is_PER:
        memoire = Memory(MEM_SIZE, env.observation_space.shape[0])
    else:
        memoire = MemoryPER(MEM_SIZE, env.observation_space.shape[0], prob_alpha=0.6, prob_beta=0.4)

    # controller (greedy)
    control = Control(action_space=env.action_space.n)

    # agent
    criterion = F.mse_loss
    optimizer = optim.Adam(params=action_value.parameters(), lr=LR ) #, weight_decay=1E-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[250, 500, 750, 1000, 1500, 3000], gamma=0.5)

    agent = AgentSimpleDQN(action_value=action_value,
                  memory=memoire,
                  control=control,
                  criterion=criterion,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  action_space=env.action_space,
                  batch_size=BATCH_SIZE,
                  gamma=GAMMA,
                  l2_lambda=L2_LAMBDA,
                  is_per=is_PER,
                  is_regul=is_regul,
                  device=device)


    trainer = Trainer(name=name, env=env, agent=agent, nb_epochs=nb_epochs, log_interval=log_interval,
                           step_valid=step_valid, nb_valid=nb_valid, render_valid=render_valid,
                            nb_test=nb_test, render_test=render_test,
                            epsilon=EPSILON, eps_min=EPSILON_MIN, eps_decay=EPSILON_DECAY
                      )

    final_score, tps_ecoule, correct = trainer.epochs()
    tab_running, tab_avg_score, tab_eval, tab_eps, tab_loss, tab_lr = trainer.get_stats()

    return final_score, tps_ecoule, correct, tab_running, tab_avg_score, tab_eval, tab_eps, tab_loss, tab_lr







