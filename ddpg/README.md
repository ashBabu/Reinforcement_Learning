### DDPG
**Notes:**
1. For the open ai gym pendulum environment, the training completes in around 80 episodes
2. The ```Loop_handler``` context manager allows you to stop the iterations 
with ```Ctrl+C``` in a nice way so that the script can carry on after the loop. Credits to [Arthur Bouton](https://github.com/Bouty92/MachineLearning) for this script and a detailed description.
3. Example of how to use the DDPG class is shown in the python notebook ```DDPG.ipynb```

<img style="float: left;" title="Episodic Rewards" src="avg_episodic_reward.png" alt="Episodic Rewards" width="300" height="300"/> <img style="float: left;" title="states and inputs" src="Trained_model_states.png" alt="states and inputs" width="300" height="300"/>
