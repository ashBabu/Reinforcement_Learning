import sys
sys.path.append('/home/ash/Ash/repo/Reinforcement_Learning/')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from ddpg import ddpg
from actor_network import Actor
import gym
import spacecraftRobot

from replaybuffer import ReplayBuffer


def reward(s, a, g):
    pass


buffer = ReplayBuffer(1e06)
# env = gym.make('SpaceRobotReach-v0')
env = gym.make('FetchReach-v1')
a_dim = env.action_space.shape[0]
s_dim = env.observation_space
# g_dim = int(s_dim/2)  # assuming state = [pos vel] and at goal, vel = 0

actor = Actor(s_dim, a_dim, s_dim).model()


# Simply wrap the goal-based environment using FlattenDictWrapper
# and specify the keys that you would like to use.
# from gym.wrappers.dict import FlattenDictWrapper
# from gym.wrappers.filter_observation import FilterObservation
# from gym.wrappers.flatten_observation import FlattenObservation
# env1 = FlattenObservation(env)
# print(env1.reset())
#
# array([ 7.6385075e-01, -1.1962976e+00,  4.7689834e+00,  2.1453280e-07,
#         4.8156938e-08, -4.6824201e-07, -5.7794364e-06,  1.2552538e-05,
#        -3.7468733e-06, -7.0022166e-09,  7.6385075e-01, -1.1962976e+00,
#         4.7689834e+00,  7.8520197e-01, -1.1049892e+00,  4.6194577e+00],
#       dtype=float32)


# env = gym.wrappers.FlattenDictWrapper(
#     env, dict_keys=['observation', 'desired_goal'])
#
# # From now on, you can use the wrapper env as per usual:
# ob = env.reset()
# print(ob.shape)  # is now just an np.array