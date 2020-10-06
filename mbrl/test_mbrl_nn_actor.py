import numpy as np
import gym
import time
import tensorflow as tf
import tensorflow.keras.optimizers as opt
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization
from tensorflow.keras import Model
from nn_dyn_learn import LearnModel


class Actor:
    def __init__(self, state_dim, action_dim, action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_range = action_bound_range

    def model(self):
        # state = Input(shape=self.state_dim, dtype='float32')
        # x = Dense(40, activation='relu', kernel_initializer=RU(-1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim)))(
        #     state)
        # x = Dense(300, activation='relu', kernel_initializer=RU(-1/np.sqrt(400), 1/np.sqrt(400)))(x)
        state = Input(shape=self.state_dim)
        x = Dense(400, activation='relu')(state)
        x = Dense(400, activation='relu')(x)
        x = BatchNormalization()(x)
        out = Dense(self.action_dim, activation='tanh', kernel_initializer=RU(-0.003, 0.003))(x)
        return Model(inputs=state, outputs=out)


if __name__ == '__main__':
    N = 50
    env_name = 'Pendulum-v0'  # obs = [cos(theta), sin(theta), thetaDot]
    actor_lr = 0.001
    minibatch_size = 50
    update_freq = 64
    buffer_size = 1e06
    n_rollouts, horizon = 40, 40

    learn_dyn = LearnModel(env_name=env_name, actor_lr=0.001, minibatch_size=50, update_freq=64, buffer_size=1e06)
    learn_dyn.dyn.load_weights('training/pend_fwd_dyn_model_64')
    """
    learn_dyn.dyn is the forward dynamics model which gives the next observation given the current 
    observation and action. For Pendulum environment, actions and obs are one dimensional and hence
    reshaping is required (obs[None, :] = obs.reshape(1, -1))
    
    next_obs = learn_dyn.dyn([obs[None, :], act[None, :]]) 
    
    """
    learn_dyn.check_model()

    env = gym.make(env_name)
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    actor = Actor(s_dim, a_dim).model()
    act_opt = opt.Adam(learning_rate=actor_lr)

    obs = env.reset()
    s = actor(obs[None, :])  # obs[None, :] = obs.reshape(1, -1)

    print('hi')





