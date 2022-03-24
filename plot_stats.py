# helper module to plot training/testing datas
import random

import matplotlib.pyplot as plt

# running score, avg score, evaluation score
# epsilon
import numpy as np


def plotScores(running, avg, eval, eps, loss, lr, name):
    fig, axs = plt.subplots(2,3, figsize=(12, 4)) #, sharey=True)

    t = np.arange(len(loss))

    axs[0][0].plot(running)
    axs[0][0].set_title("Running Score")
    axs[0][1].plot(avg)
    axs[0][1].set_title("Average score")
    axs[0][2].plot(eval)
    axs[0][2].set_title("Evaluation score")

    axs[1][0].plot(loss)
    axs[1][0].set_title("Loss")
    axs[1][1].plot(eps)
    axs[1][1].set_title("Epsilon")
    axs[1][2].plot(lr)
    axs[1][2].set_title("LR")


    plt.plot(eps)
    fig.suptitle('SCORES')
    #fig.savefig(f"./FIG/{name}.png")

    plt.show()
"""
import pickle

with open("./DATA/SimpleDQN_LRSched128_2000_lr0.001_batch1024_g0.75.res", "rb") as fp:
    res = pickle.load(fp)



#res = {'running':running, "avg_score":avg_score, "eval":eval,
#       "eps":eps, "loss":loss, "final_score":final_score,
#       "tps_ecoule":tps_ecoule, "correct":correct
#       }

plotScores(name="", running=res['running'], avg=res['avg_score'], eval=res['eval'], eps=res['eps'], loss=res['loss'], lr=res['lr'])
"""