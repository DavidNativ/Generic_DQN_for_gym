# Generic_DQN_for_gym
Generic modules for solving gym discrete env

The goal of this project is to create a modular agent that could solve gymAI discrete environments - in this case it has been tested on LunarLander and CartPole. 
The basic architecture is:

- Neural Network: for evaluating Q action values
- Memory: a replay buffer that implements "Experience Replay". The idea is provide batches to the model for training, batches that are not correlated. I chose to store the tuples (state, action, reward, next step, done flag) in distinct numpy arrays, for the code is much easier to read than with a deque, and the performances are supposed to be much better
- Control: a class that implements the controler, ie. determine the action to take according to a state and a policy

- Trainer is a class that is used to instanciate trainings of the model and its parameters

The work is still on progress, even if the code already runs. A lot of improvements are to be done in the structure of the project, in order to make it clearer and lighter

NB. Modules architecture allows to try different approches; thus I also implemented a PER Memory, that ponder the experiences, and a Double DQN. The latter uses two distincts NN, one for evaluating the target and one for getting the action value. This brings stability to the algorithm. The target network is updated with the action value network every n steps

Files structure
---------------
- main : the main file, that instanciate the agent, trains it and plot the results
- AgentDQN : the agent, that groups the modules (self contains the NN, and instances of Memory and Control)
- Trainer: drives the training

for the Double DQN:
- AgentDDQN

Note that the Memory module can be instanciated either in a simple way (Memory.py) or with PER principe (PERMemory.py)

Improvements
------------
- create a clearer flow, with a dedicated main function that would permit to select the type of agent
- create an arguments parser for calling the main function directly from command line
- use Abstract Classes / Inheritance for the more complex modules


