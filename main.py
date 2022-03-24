from plot_stats import plotScores
from AgentDQN import *
from AgentDDQN import *
import pickle

EPSILON = 1
EPSILON_DECAY = 4E-4
EPSILON_MIN = 0.1

L2_LAMBDA = 1E-5
MEM_SIZE = 1000000

hidden_size=256

nb_epochs=4000
log_interval=20

step_valid=200
nb_valid=20
render_valid=False

nb_test=100
render_test=False

IS_REGUL = False
IS_PER = False

if __name__ == "__main__":

    #===================#
    #  BENCH DDQN       #
    #===================#
    nb_epochs=4000
    log_interval=40

    step_valid=100
    nb_valid=20
    render_valid=False
    LR = 1E-3
    GAMMA = 0.75
    BATCH_SIZE = 256


    EPSILON = 1
    EPSILON_DECAY = 4E-4
    EPSILON_MIN = 0.1

    L2_LAMBDA = 1E-5
    ALPHA = 0.8
    BETA = 0.4

    FREQ_UPDATE_TARGET = -1
    tab_eval = []

    #for i, freq in enumerate([10, 5, 2]):
        #FREQ_UPDATE_TARGET  = freq
    name = f"DDQN_LRSched{hidden_size}_{nb_epochs}_lr{LR}_batch{BATCH_SIZE}_g{GAMMA}_{FREQ_UPDATE_TARGET}"
    if IS_PER:
        name += "_PER"
    if IS_REGUL:
        name += "_L2"
    final_score, tps_ecoule, correct, running, avg_score, eval, eps, loss, lr = DDQN(name,
                            LR, BATCH_SIZE, GAMMA,
                            EPSILON, EPSILON_DECAY, EPSILON_MIN, L2_LAMBDA,
                            MEM_SIZE, hidden_size, FREQ_UPDATE_TARGET,
                            nb_epochs, log_interval,
                            step_valid, nb_valid, render_valid,
                            nb_test, render_test,
                            ALPHA, BETA,
                            IS_PER, IS_REGUL
                            )
    tab_eval.append([name, final_score, tps_ecoule, correct, lr])

    res = {'running':running, "avg_score":avg_score, "eval":eval, "eps":eps,
           "loss":loss, "lr":lr, "final_score":final_score, "tps_ecoule":tps_ecoule, "correct":correct}

    with open(f"./DATA/{name}.res", "wb") as fp:
        pickle.dump(res, fp)


    import matplotlib.pyplot as plt

    print(tab_eval)
    plotScores(res['running'], res['avg_score'], res['eval'], res['eps'], res['loss'], res['lr'], name )

    """
    fig, axs = plt.subplots(12,3, figsize=(12, 4)) #, sharey=True)
    t = np.arange(len(loss))
    i, j = 0, 0
    for name_, correct_, final_, tps_ecoule in tab_eval:
        with open(f"./DATA/{name_}.res", "wb") as fp:
            res = pickle.load(fp)
            axs[i][0].set_title(res['name'])
            axs[i][0].plot(res['running'])
            axs[i][1].plot(res['avg'])
            axs[i][2].plot(res['final_score'])
            i = (i + 1)
    
    plt.show()
    """

    """
    #===================#
    #  BENCH SimpleDQN  #
    #===================#
    EPSILON = 1
    EPSILON_DECAY = 1E-3
    EPSILON_MIN = 0.05

    L2_LAMBDA = 1E-5
    MEM_SIZE = 100000

    log_interval=20

    step_valid=200
    nb_valid=20
    render_valid=False

    nb_test=100
    render_test=False

    IS_REGUL = False
    IS_PER = False

    tab_eval = []
    i = 0
    for LR in [1E-3, 5E-4, 1E-4]:
        for GAMMA in [0.75, 0.99]:
            for BATCH_SIZE in [64, 128]:
                name = f"{i}_SimpleDQN{hidden_size}_{nb_epochs}_lr{LR}_batch{BATCH_SIZE}_g{GAMMA}"
                if IS_PER:
                    name += "_PER"
                if IS_REGUL:
                    name += "_L2"
                final_score, tps_ecoule, correct, running, avg_score, eval, eps, loss = DQL(name,
                                        LR, BATCH_SIZE, GAMMA,
                                        EPSILON, EPSILON_DECAY, EPSILON_MIN, L2_LAMBDA,
                                        MEM_SIZE, hidden_size,
                                        nb_epochs, log_interval,
                                        step_valid, nb_valid, render_valid,
                                        nb_test, render_test,
                                        IS_PER, IS_REGUL
                                        )
                tab_eval.append([name, final_score, tps_ecoule, correct])

                res = {'running':running, "avg_score":avg_score, "eval":eval, "eps":eps, "loss":loss, "final_score":final_score, "tps_ecoule":tps_ecoule, "correct":correct}
                with open(f"./DATA/{name}.res", "wb") as fp:
                    pickle.dump(res, fp)

                i += 1


    print(tab_eval)


    fig, axs = plt.subplots(4,3, figsize=(12, 4)) #, sharey=True)
    t = np.arange(len(loss))
    i, j = 0, 0
    for name_, correct_, final_, tps_ecoule in tab_eval:
        with open(f"./DATA/{name_}.res", "wb") as fp:
            res = pickle.load(fp)
            axs[i][j].plot(res['eval'])
            axs[i][j].set_title(res['name'])
            i = (i + 1) // 4
            j = (j + 1) % 4
    plt.show()


    #RESULTATS
    #[['0_SimpleDQN128_2000_lr0.001_batch64_g0.75', 27.0, 1558.97, 25], 
    ['0_SimpleDQN128_2000_lr0.001_batch128_g0.75', -24.0, 1409.79, 24], 
    ['0_SimpleDQN128_2000_lr0.001_batch64_g0.99', -270.0, 756.85, 0], 
    ['0_SimpleDQN128_2000_lr0.001_batch128_g0.99', -176.0, 1345.07, 0], 
    ['1_SimpleDQN128_2000_lr0.0005_batch64_g0.75', -6.0, 1372.15, 23], 
    ['1_SimpleDQN128_2000_lr0.0005_batch128_g0.75', 40.0, 1256.88, 32], 
    ['1_SimpleDQN128_2000_lr0.0005_batch64_g0.99', -127.0, 996.26, 0], 
    ['1_SimpleDQN128_2000_lr0.0005_batch128_g0.99', -152.0, 1097.16, 1], 
    ['2_SimpleDQN128_2000_lr0.0001_batch64_g0.75', 18.0, 1462.03, 21], 
    ['2_SimpleDQN128_2000_lr0.0001_batch128_g0.75', 50.0, 1512.53, 36], 
    ['2_SimpleDQN128_2000_lr0.0001_batch64_g0.99', -100.0, 1473.79, 0], 
    ['2_SimpleDQN128_2000_lr0.0001_batch128_g0.99', -212.0, 1550.73, 0]]
    """

    ##############################################################################

    """
    #===================#
    #  BENCH PER / L2   #
    #===================#
    tab_eval = []
    i = 0

    LR = 1E-4
    GAMMA = 0.75
    BATCH_SIZE = 128

    for IS_PER in [True, False]:
        for IS_REGUL in [True, False]:

            name = f"{i}_SimpleDQN{hidden_size}_{nb_epochs}_lr{LR}_batch{BATCH_SIZE}_g{GAMMA}"
            if IS_PER:
                name += "_PER"
            if IS_REGUL:
                name += "_L2"
            final_score, tps_ecoule, correct, running, avg_score, eval, eps, loss = DQL(name,
                                    LR, BATCH_SIZE, GAMMA,
                                    EPSILON, EPSILON_DECAY, EPSILON_MIN, L2_LAMBDA,
                                    MEM_SIZE, hidden_size,
                                    nb_epochs, log_interval,
                                    step_valid, nb_valid, render_valid,
                                    nb_test, render_test,
                                    IS_PER, IS_REGUL
                                    )
            tab_eval.append([name, final_score, tps_ecoule, correct])

            res = {'running':running, "avg_score":avg_score, "eval":eval, "eps":eps, "loss":loss, "final_score":final_score, "tps_ecoule":tps_ecoule, "correct":correct}
            with open(f"./DATA/{name}.res", "wb") as fp:
                pickle.dump(res, fp)

            i += 1


    print(tab_eval)

    res
    [['0_SimpleDQN128_2000_lr0.0001_batch128_g0.75_PER_L2', -101.0, 1616.67, 1], 
    ['1_SimpleDQN128_2000_lr0.0001_batch128_g0.75_PER', -88.0, 1581.58, 8],
     ['2_SimpleDQN128_2000_lr0.0001_batch128_g0.75_L2', -30.0, 1654.41, 4],
      ['3_SimpleDQN128_2000_lr0.0001_batch128_g0.75', -20.0, 1360.77, 22]]
    """














