import numpy as np
import gym
import spacecraftRobot

# env = gym.make('FetchReach-v1')
# env = gym.make('SpaceRobotPickAndPlace-v0')
env = gym.make('SpaceRobotReach-v0')
# env = gym.make('SpaceRobot-v0')
obs = env.reset()
done = False


def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()


while not done:
    action = policy(obs['observation'], obs['desired_goal'])
    obs, reward, done, info = env.step(action)
    env.render()
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(
        obs['achieved_goal'], substitute_goal, info)
    print('reward is {}, substitute_reward is {}'.format(
        reward, substitute_reward))


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