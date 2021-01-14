# Double DQN RL for Vehicle Overtake using CNNs
<img src="files/media/image20.gif" align="right" width="400" height="200">

This Repo contains code and instructions for implementing Double DQN Reinforcemnt Learning in an OpenAI Gym like environment. It takes image as the input and the action space as the output for the network. 

***Idea:** The main idea is to learn how to do an overtaking action using Double DQN RL in a top-down view highway environment.
Since this project uses an OpenAI Gym like format, it will be easy for anyone to modify this project their gym like environment within seconds.*

## Requirements
* Python 3.7
* Tensorflow 2.3.0
* Numpy
* OpenCV
* Your OpenAI-Gym like Environment
* In this case, get [Highway Environment](https://github.com/eleurent/highway-env)

## Deep Reinforcement Learning


## Model Architecture
<img src="files/media/image11.png" width="800" height="300">


## Double DQN
<img align="center">

<img src="files/media/Double DQN.png" width="800" height="400">
</p>  

<img align="center" src="files/media/image3.png" width="500" height="200">



## Training

## Online Training

Our network consists of two main parts, which is the training network and the predicting network. As we have discussed in the double deep Q learning section, these are our two estimators. The pseudo code of our implementation can seen below.

The figure below shows how the overview of our network would be, the training network will train on the data and gather the best parameters for the model. These parameters are then sent to the predicting network, which will execute these actions on the state and then send the results back to the training network. The training network will further use this data to tweak its parameters and train again. By doing this operation of leveling up, we teach the agent how to overtake depending on the state and the actions.



### Loss
<img src="files/media/image2.png" align="right" width="400" height="350">

The loss graph will show that the network is learning better after each epoch as it shows the overall the network is making better decisions. The loss graph is drawn for the training network. The loss graph is much smoother after we reduced the learning rate to a very small value and we have also smoothend the loss value using Tensorboard. The following graph is what we got for the final iteration of our implementation. 

### Reward Graph
<img src="files/media/image5.png" align="right" width="400" height="350">

The rewards graph will show that as the epochs keep increasing the rewards will also increase because as the network learns more, it will perform actions which gives it the maximum reward. The original paper where they developed the Double DQN technique, the authors ran the environment for 250M epochs compared to us, we only ran the training for 250K times.

As we have mentioned above, the accuracy graph is not a good measure of how well our network performs in a reinforcement learning task but, as you can see in the loss graph, the loss decreases with each epoch which suggests that the network is learning properly. 

## How to Run 
## Clone this Repository:  

    git clone https://github.com/perseus784/rl_overtake.git

Download [this google drive folder](https://drive.google.com/drive/folders/1_p5Pcj7jhFgoOf-L-QueEpoTMoTW-C34?usp=sharing) and put the *training* folder inside the files folder.



      python run.py
This will start the training but make sure that you delete the previous training data like models, tensorboard, plotting csvs from the logs folders. See [here](https://youtu.be/akqh1cmFD-k) for how to run.

### Command to install
\begin{itemize}
    \item Highway Environment: pip install --user
    \item git+https: github.com/eleurent/highway-env
    \item Python: https://www.python.org/downloads/
    \item Tensorflow: pip3 install --user tensorflow-gpu
\end{itemize}


## Results

We built a double deep Q learning network and ran the network on the highway environment. After reiterating for multiple times to get the hyperparamters right, we finally got aversion which worked. We can see the loss and reward graphs of our network below. 


\subsection{Links}
We have created seperate external links for each of the things below,

\begin{itemize}
    \item Demo: \url{https://youtu.be/sH00TWLwBoA}
    \item How to Run: \url{https://youtu.be/akqh1cmFD-k}
    \item Project: \url{https://github.com/perseus784/rl_overtake} (This is a private repository, please request for access if you need to access it)
    \item Trained models and Tensorboard metrics can be found in this drive folder: \url{https://drive.google.com/drive/folders/1_p5Pcj7jhFgoOf-L-QueEpoTMoTW-C34?usp=sharing}
\end{itemize}
You can use these external links to see the performance of the model in real time application. Specific instruction on how to run the code for training or testing is given the readme file attached below after the code.


## To test:
Open command prompt on folder rl_overtake and do *python [trail_run.py](https://github.com/perseus784/rl_overtake/blob/master/trail_run.py)* and you can see the agent running in the highway environment after loading the restored model. [Demo](https://youtu.be/sH00TWLwBoA)

<p float = "left">
<img src="files/media/image14.gif" width="400" height="200">
</p> 

