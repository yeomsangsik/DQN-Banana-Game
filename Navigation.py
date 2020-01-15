#!/usr/bin/env python
# coding: utf-8

# # Navigation
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# In[2]:


get_ipython().system('pip -q install ./python')


# The environment is already saved in the Workspace and can be accessed at the file path provided below.  Please run the next code cell without making any changes.

# In[3]:


from unityagents import UnityEnvironment
import numpy as np
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('python -m pip install pyvirtualdisplay')

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# please do not modify the line below
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[4]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

# In[5]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Note that **in this coding environment, you will not be able to watch the agent while it is training**, and you should set `train_mode=True` to restart the environment.

# In[6]:


from yeom_agent import Agent

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))


# In[6]:


def DQN_by_yeom(n_episode = 3000 , max_t =500, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995):
    
    scores=[]
    scores_window = deque (maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episode+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            #next_state, reward, done, _ = env.step(action)[brain_name]
            
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward            
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores    

scores = DQN_by_yeom()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# In[32]:


agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

score = 0                                              # initialize the score
test_episode_num = 100 
for i in range(test_episode_num):
    env_info = env.reset()[brain_name]                 # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    
                                             
    for y in range(200):
        action = agent.act(state, 0.0)                 # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
               
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
  
print("Score: {}".format(score))


# When finished, you can close the environment.

# In[33]:


env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
# - When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# - To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
# - In this coding environment, you will not be able to watch the agent while it is training.  However, **_after training the agent_**, you can download the saved model weights to watch the agent on your own machine! 
