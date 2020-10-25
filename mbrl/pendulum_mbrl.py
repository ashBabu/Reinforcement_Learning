"""
Same as approximate dynamics, but now the input is sine and cosine of theta (output is still dtheta)
This is a continuous representation of theta, which some papers show is easier for a NN to learn.
"""

import gym
import random
import numpy as np
import torch
import logging
import math
import time
import mppi_fast
import tensorflow as tf
from gym import wrappers, logger as gym_log

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 15  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0
    nx = 2  # state_dimension
    nu = 1 # action dimension

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    noise_sigma = torch.tensor(1, device=d, dtype=dtype)
    lambda_ = 1.

    randseed = 24
    if randseed is None:
        randseed = random.randint(0, 1000000)
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    logger.info("random seed %d", randseed)

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    def train(dataset):
        """
        Trying to find the increment in states, f_(theta), from the equation
        s_{t+1} = s_t + dt * f_(theta)(s_t, a_t)
        """
        dtheta = angular_diff_batch(dataset[1:, 0], dataset[:-1, 0])
        dtheta_dt = dataset[1:, 1] - dataset[:-1, 1]
        Y = np.hstack((dtheta.reshape(-1, 1), dtheta_dt.reshape(-1, 1)))  # x' - x residual
        xu = dataset[:-1]  # make same size as Y
        xu = np.hstack((np.cos(xu[:, 0]).reshape(-1, 1), np.sin(xu[:, 0]).reshape(-1, 1), xu[:, 1:]))
        dyn_network.fit(xu, Y, epochs=100)

    def dynamics(state, perturbed_action):
        state, perturbed_action = state.numpy(), perturbed_action.numpy()
        u = np.clip(perturbed_action, ACTION_LOW, ACTION_HIGH)
        obs = state2obs(state)
        xx = np.hstack((obs, u))
        state_residual = dyn_network(xx)
        # output dtheta directly so can just add
        next_state = state + state_residual.numpy()
        return torch.from_numpy(next_state)

    def true_dynamics(state, perturbed_action):
        # true dynamics from gym
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((newth, newthdot), dim=1)
        return state

    def angular_diff_batch(a, b):
        """Angle difference from b to a (a - b)"""
        d = a - b
        d[d > math.pi] -= 2 * math.pi
        d[d < -math.pi] += 2 * math.pi
        return d

    def angle_normalize(x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    def running_cost(state, action):
        theta = state[:, 0]
        theta_dt = state[:, 1]
        action = action[:, 0]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2 + 0.001 * action ** 2
        return cost

    def dyn_model(in_dim, out_dim):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(in_dim, )),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(out_dim),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    in_dim = nx + nu + 1
    out_dim = nx

    dyn_network = dyn_model(in_dim, out_dim)
    # dyn_network.save_weights('pend_fwd_dyn_weights')

    def obs2state(obs):
        a = np.array([np.arctan2(obs[1], obs[0]), obs[2]])
        a[0] = angle_normalize(a[0])
        return a

    def state2obs(state):
        state = np.array(state)
        if state.ndim > 1:
            return np.array([np.cos(state[:, 0]), np.sin(state[:, 0]), state[:, 1]]).T
        else:
            return np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

    downward_start = True
    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = [np.pi, 1]

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        logger.info("bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = env.state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
            env.step([action])
            # env.render()
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

        train(new_data)
        logger.info("bootstrapping finished")

    env = wrappers.Monitor(env, '/tmp/mppi/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]

    mppi_gym = mppi_fast.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                         u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
    total_reward, data = mppi_fast.run_mppi(mppi_gym, env, train)
    logger.info("Total reward %f", total_reward)
