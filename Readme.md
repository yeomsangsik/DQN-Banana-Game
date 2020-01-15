I have successfully completed the Udacy Term project.

In this project, you will be able to train your agents using DQN method. 

The technique I used was the initial DQN proposed by DeepMind team in 2013.






file:///home/yss/Desktop/Screenshot%20from%202020-01-13%2017-38-28.png






[state, action, reward]


State space dimension : 37 (the agent's velocity, along with ray-based perception of objects around agent's forward direction.)  
Action space dimension : 4 (Move forward,back,right,left)

Reward : +1 if the agent collected a yellow banana
         -1 if the agent collected a blue banana  

[Deploying method]

First, I made model and agent files for easy analysis of modules.

The existing DQN code format was brought back to apply the state and action variables required by Banana game.

Second, to make full use of the experience replay method, the max time step value was set up to increase the effect of learning. The difference (average score) between 200 and 500 for max time step was more than twice.

Implemented modules torchIt was saved using save() so that it can be used in other workspaces

[test]

I called in the module and tested the episode 100 times. As a result, the average value was more than 10.0 but slightly lower than the average return on learning was 13.0. 

[Environment install]

Step 1: Clone the DRLND Repository
If you haven't already, please follow the instructions in the DRLND GitHub repository to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

Step 2: Download the Unity Environment

you can download the Unity Environment from Unity homepage links. You need select the environment that matches your operating system.

Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)