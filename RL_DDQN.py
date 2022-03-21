#=============================================#
# GENERAL REINFORCEMENTDeep Q Learning AGENT  #
#=============================================#

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import gym

from Memory import Memory
from MemoryPER import MemoryPER

from Control import Control
from SimpleDQN import SimpleDQN
from AgentDDQN import AgentDDQN
from Trainer import Trainer


#main class: the agent acts, remembers and learns
# -----------------------------------------------
def DDQN(name, LR, BATCH_SIZE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN, L2_LAMBDA,
        MEM_SIZE, hidden_size, freq_update_target,
        nb_epochs, log_interval,
        step_valid, nb_valid, render_valid,
        nb_test, render_test,
        alpha=0.6, beta=0.4,
        is_PER=False, is_regul=False
        ):

    print( "\n#=========================#==========================#\n"
           "# GENERAL REINFORCEMENT Double Deep Q Learning AGENT #\n"
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
    action_value = SimpleDQN(input_size=env.observation_space.shape[0],
                             hidden_size_in=hidden_size, hidden_size_out=hidden_size,
                             output_size=env.action_space.n).to(device)

    target = SimpleDQN(input_size=env.observation_space.shape[0],
                             hidden_size_in=hidden_size, hidden_size_out=hidden_size,
                             output_size=env.action_space.n).to(device)

    #m = "./MODELS/DDQN_LRSched128_4000_lr0.001_batch64_g0.99_-1_L2_2022-2-19_22-4-18_2000-4000"
    #action_value.load(m)
    #target.load(m)

    # Replay Buffer
    if not is_PER:
        memoire = Memory(MEM_SIZE, env.observation_space.shape[0])
    else:
        memoire = MemoryPER(MEM_SIZE, env.observation_space.shape[0], prob_alpha=alpha, prob_beta=beta)

    # controller (greedy)
    control = Control(action_space=env.action_space.n)

    # agent
    criterion = F.mse_loss
    optimizer = optim.Adam(params=action_value.parameters(), lr=LR ) #, weight_decay=1E-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500, 2000, 2500, 3000], gamma=0.75)

    agent = AgentDDQN(action_value=action_value,
                  target=target,
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
                  freq_update_target=freq_update_target,
                  device=device)


    trainer = Trainer(name=name, env=env, agent=agent, nb_epochs=nb_epochs, log_interval=log_interval,
                           step_valid=step_valid, nb_valid=nb_valid, render_valid=render_valid,
                            nb_test=nb_test, render_test=render_test,
                            epsilon=EPSILON, eps_min=EPSILON_MIN, eps_decay=EPSILON_DECAY
                      )

    final_score, tps_ecoule, correct = trainer.epochs()
    tab_running, tab_avg_score, tab_eval, tab_eps, tab_loss, tab_lr = trainer.get_stats()

    return final_score, tps_ecoule, correct, tab_running, tab_avg_score, tab_eval, tab_eps, tab_loss, tab_lr







