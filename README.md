# rl_overtake

## Install 
* Python 3.7
* Tensorflow 2.3.0
* Numpy
* OpenCV
* Highway Environment from: https://github.com/eleurent/highway-env 


<img align="center" src="files/media/image3.png" width="500" height="200">

## Clone this Repository:  

    git clone https://github.com/perseus784/rl_overtake.git

Download [this google drive folder](https://drive.google.com/drive/folders/1_p5Pcj7jhFgoOf-L-QueEpoTMoTW-C34?usp=sharing) and put the *training* folder inside the files folder.

## To test:
Open command prompt on folder rl_overtake and do *python [trail_run.py](https://github.com/perseus784/rl_overtake/blob/master/trail_run.py)* and you can see the agent running in the highway environment after loading the restored model. [Demo](https://youtu.be/sH00TWLwBoA)

<p float = "left">
<img src="files/media/image14.gif" width="400" height="200">
<img src="files/media/image20.gif" width="400" height="200">
</p> 

## To run the training:  

      python run.py
This will start the training but make sure that you delete the previous training data like models, tensorboard, plotting csvs from the logs folders. See [here](https://youtu.be/akqh1cmFD-k) for how to run.


## Network Architecture:
<img align="center">
<img src="files/media/Double DQN.png" width="700" height="400">
</p> 

## Loss and Rewards:

<p float = "left">
<img src="files/media/image2.png" width="400" height="350">
<img src="files/media/image5.png" width="400" height="350">
</p> 
