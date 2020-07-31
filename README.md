# Reinforcement_Learning
Implementation of various RL algorithms in Tensorflow 2.x

### DDPG
**Notes:**
1. For the open ai gym pendulum environment, the training completes in around 80 episodes
2. The ```looptools.py``` is used to handle sudden interruptions (Ctrl+C). This saves the model 
at the current iteration
3. Example of how to use the DDPG class is shown in ```ddpg.py``` in the ```if __name__ == '__main__' ``` function

<img style="float: left;" title="Episodic Rewards" src="ddpg/avg_episodic_reward.png" alt="Episodic Rewards" width="400" height="400"/>

<img style="float: left;" title="states and inputs" src="ddpg/Trained_model_states" alt="states and inputs" width="400" height="400"/>

