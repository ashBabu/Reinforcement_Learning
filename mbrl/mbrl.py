import gym
import random
import numpy as np
import torch
import logging
import math
import mppi_fast
import tensorflow as tf
from gym import wrappers, logger as gym_log
import copy

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


class MBRL:
    def __init__(self, env_name, horizon=15, rollouts=100, epochs=150,
                bootstrapIter=100, bootstrap=True, noise_sigma=1, lambda_=1, downward_start=True):
        self.env = gym.make(env_name)
        self.env.reset()
        self.env_cpy = copy.deepcopy(self.env.env)  # used only when true_dynamics_gym is used
        self.horizon = horizon
        self.rollouts = rollouts
        self.a_low, self.a_high = self.env.action_space.low, self.env.action_space.high
        self.s_dim, self.a_dim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        """dyn_model has s_dim+a_dim-1 because s_dim = 3 for pendulum (sin(theta), cos(theta), thetadot). Here 
        we consider that we are dealing with states ie., theta and thetadot"""
        self.fwd_dyn_nn = self.dyn_model(self.s_dim+self.a_dim-1, self.s_dim-1)
        self.d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.double
        self.noise_sigma = torch.tensor(noise_sigma, device=self.d, dtype=self.dtype)
        self.lambda_ = lambda_
        self.epochs = epochs,
        self.bootstrapIter = bootstrapIter
        if downward_start:
            self.downwardstart()
        if bootstrap:
            self.bootstrap()

        randseed = 24
        if randseed is None:
            randseed = random.randint(0, 1000000)
        random.seed(randseed)
        np.random.seed(randseed)
        torch.manual_seed(randseed)
        logger.info("random seed %d", randseed)

        self.mppi_gym = mppi_fast.MPPI(self.dynamics, self.running_cost, self.s_dim-1, self.noise_sigma, num_samples=self.rollouts,
                                       horizon=self.horizon, lambda_=self.lambda_, device=self.d,
                                  u_min=torch.tensor(self.a_low, dtype=torch.double, device=self.d),
                                  u_max=torch.tensor(self.a_high, dtype=torch.double, device=self.d))

    def run_mbrl(self):
        total_reward, data = mppi_fast.run_mppi(self.mppi_gym, self.env, self.train)

    def bootstrap(self):
        logger.info("bootstrapping with random action for %d actions", self.bootstrapIter)
        state_dim = self.s_dim - 1  # for pendulum
        new_data = np.zeros((self.bootstrapIter, state_dim+self.a_dim))
        for i in range(self.bootstrapIter):
            pre_action_state = self.env.env.state
            action = np.random.uniform(low=self.a_low, high=self.a_high)
            self.env.step([action])
            # env.render()
            new_data[i, :state_dim] = np.squeeze(pre_action_state)
            new_data[i, state_dim:] = action

        self.train(new_data)
        logger.info("bootstrapping finished")
        self.downwardstart()

    def downwardstart(self):
        self.env.env.state = np.array([np.pi, 1])

    def train(self, dataset):
        """
        Trying to find the increment in states, f_(theta), from the equation
        s_{t+1} = s_t + dt * f_(theta)(s_t, a_t)
        """
        dataset[:, 0] = self.angle_normalize(dataset[:, 0])
        d = self.s_dim - 1
        a = int(d/2) - 1
        dtheta = self.angular_diff_batch(dataset[1:, a], dataset[:-1, a])
        dtheta_dt = dataset[1:, a+1:d] - dataset[:-1, a+1:d]
        Y = np.hstack((dtheta.reshape(-1, 1), dtheta_dt.reshape(-1, 1)))  # x' - x residual
        xu = dataset[:-1]  # make same size as Y
        # xu = np.hstack((np.cos(xu[:, 0]).reshape(-1, 1), np.sin(xu[:, 0]).reshape(-1, 1), xu[:, 1:]))
        self.fwd_dyn_nn.fit(xu, Y, epochs=150)

    def dynamics(self, state, perturbed_action):
        state, perturbed_action = state.numpy(), perturbed_action.numpy()
        u = np.clip(perturbed_action, self.a_low, self.a_high)
        xx = np.hstack((state, u))
        state_residual = self.fwd_dyn_nn(xx)
        # output dtheta directly so can just add
        next_state = state + state_residual.numpy()
        next_state[:, 0] = self.angle_normalize(next_state[:, 0])
        return torch.from_numpy(next_state)

    def true_dynamics_gym(self, state, perturbed_action):
        """
         when using this function as input argument to mppi, comment out the line retrain_dynamics(dataset) in run_mppi
        """
        ss = state.shape[0]
        next_state = np.zeros_like(state)
        u = torch.clamp(perturbed_action, -2, 2)
        for i in range(ss):
            self.env_cpy.state = state[i]
            self.env_cpy.step(u[i])
            next_state[i] = self.env_cpy.state
        return torch.from_numpy(next_state)

    def angular_diff_batch(self, a, b):
        """Angle difference from b to a (a - b)"""
        d = a - b
        d[d > math.pi] -= 2 * math.pi
        d[d < -math.pi] += 2 * math.pi
        return d

    def angle_normalize(self, x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    def running_cost(self, state, action):
        theta = state[:, 0]
        theta_dt = state[:, 1]
        action = action[:, 0]
        cost = self.angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2 + 0.001 * action ** 2
        return cost

    def dyn_model(self, in_dim, out_dim):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(in_dim, )),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(out_dim),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def obs2state(self, obs):
        a = np.array([np.arctan2(obs[1], obs[0]), obs[2]])
        a[0] = self.angle_normalize(a[0])
        return a

    def state2obs(self, state):
        state = np.array(state)
        if state.ndim > 1:
            return np.array([np.cos(state[:, 0]), np.sin(state[:, 0]), state[:, 1]]).T
        else:
            return np.array([np.cos(state[0]), np.sin(state[0]), state[1]])


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 15  # T
    N_SAMPLES = 100  # K

    mbrl = MBRL(ENV_NAME)
    mbrl.run_mbrl()
    # env = wrappers.Monitor(env, '/tmp/mppi/', force=True)
    # env.reset()
