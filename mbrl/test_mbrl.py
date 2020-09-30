import numpy as np
import gym
import time
from nn_dyn_learn import LearnModel


class MPPI:
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, nn_dyn_model, env_name='Pendulum-v0', K=200, T=20, gamma=0.98, U=None, lambda_=1.0,
                 noise_mu=0, noise_sigma=1, u_init=1, render=False, downward_start=True, save=True):
        """
        :param nn_dyn_model: neural network which takes in [states, action] and gives [next_state]. Or it is a
                function approximator for the forward dynamics model of a robot
        :param K: Number of rollouts
        :param T: MPC's prediction horizon
        :param gamma:
        :param U:
        :param lambda_:
        :param noise_mu:
        :param noise_sigma:
        :param u_init:
        :param downward_start: Only valid for 'Pendulum-v0' environment.
                               if True, then the pendulum starts from vertically downward position
        :param save:
        """
        self.fwd_dyn = nn_dyn_model
        self.env = gym.make(env_name)
        self.env.reset()
        self.render = render
        self.rollouts = K  # N_SAMPLES
        self.horizon = T  # TIMESTEPS
        self.gamma = gamma
        self.a_dim = self.env.action_space.shape[0]
        self.max_torque = self.env.action_space.high
        self.min_torque = self.env.action_space.low

        """ To set initial guess value of U """
        if not U:
            low = self.env.action_space.low
            high = self.env.action_space.high
            self.U = np.squeeze(np.random.uniform(low=low, high=high, size=(self.horizon, self.a_dim)))
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.u_init = u_init
        self.cost_total = np.zeros(shape=self.rollouts)
        self.optimized_actions = []

        if env_name == 'Pendulum-v0' and downward_start:
            self.env.env.state = [np.pi, 1]

        """
        MPPI requires env.get_state() and env.set_state() function in addition to the gym env functions
        For envs like 'Pendulum-v0', there is an already existing function 'env.env.state' which can be 
        used to get and set state.
        """
        self.x_init = self.get_state()

        if save:
            np.save('initial_state', self.x_init, allow_pickle=True)

        self.noise = self.get_noise(k=self.horizon, t=self.rollouts, a_dim=self.a_dim)

        # if noise_gaussian:
        #     self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.rollouts*self.horizon, self.a_dim))
        # else:
        #     self.noise = np.full(shape=(self.rollouts, self.horizon), fill_value=0.9)

    def set_state(self, state):
        """
           This method has to be implemented for envs other than pendulum
           Refer: 'https://github.com/aravindr93/trajopt/blob/master/trajopt/envs/reacher_env.py#L87'
           The above is for  MuJoCo envs
        """
        self.env.env.state = state

    def get_state(self):
        """
        'https://github.com/aravindr93/trajopt/blob/master/trajopt/envs/reacher_env.py#L81'
        'https://github.com/ashBabu/spaceRobot_RL/blob/master/spacecraftRobot/envs/spaceRobot.py#L148'
        """
        # if env_name == 'Pendulum-v0':
        return self.env.env.state

    def get_noise(self, k=1, t=1, a_dim=1):
        return np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(t, k, a_dim))

    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    def reward(self, curr_obs, action):
        """ For pendulum, obs = [cos(theta), sin(theta), theta_dot]"""
        curr_obs = curr_obs.numpy()[0]
        th = np.arctan2(curr_obs[1], curr_obs[0])
        return self.angle_normalize(th) ** 2 + .1 * curr_obs[2] ** 2 + .001 * (action[0] ** 2)

    def state2obs(self, state):
        return np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

    def _compute_total_cost(self, state, k):
        self.set_state(state)
        cost = 0
        obs = self.state2obs(state)
        for t in range(self.horizon):
            perturbed_actions = self.U[t] + self.noise[k, t]
            # next_state, reward, done, info = self.env.step(perturbed_actions)
            actions = np.clip(perturbed_actions, self.min_torque, self.max_torque)
            next_state = self.fwd_dyn([obs[None, :], actions[None, :]])
            reward = self.reward(next_state, actions)
            cost += self.gamma ** t * -reward
        return cost

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def control(self, iter=200):
        for _ in range(iter):
            for k in range(self.rollouts):
                self.cost_total[k] = self._compute_total_cost(k=k, state=self.x_init)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1 / self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1 / eta * cost_total_non_zero

            for t1 in range(self.horizon):
                for k1 in range(self.rollouts):
                    self.U[t1] += omega[k1] * self.noise[k1, t1]
            self.set_state(self.x_init)
            # if self.U[0].ndim == 0:
            #     # s, r, _, _ = self.env.step([self.U[0]])
            #     s = self.fwd_dyn([self.x_init, self.U[0]])
            # else:
            #     # s, r, _, _ = self.env.step(self.U[0])
            obs = self.state2obs(self.x_init)
            act = np.array([self.U[0]])[None, :]
            act = np.clip(act, self.min_torque, self.max_torque)
            print('state b4', self.get_state())
            s = self.fwd_dyn([obs[None, :], act[0]])
            r = self.reward(s, act)
            self.set_state(s.numpy()[0])
            print('state after', self.get_state())
            self.optimized_actions.append(act[0][0])
            self.x_init = self.get_state()
            self.U = np.roll(self.U, -1, axis=0)  # shift all elements to the left
            self.U[-1] = 1.  #
            print("iter no: %0.2f action taken: %.2f cost received: %.2f" % (_, act[0][0], -r))
            if self.render:
                self.env.render()
            self.cost_total[:] = 0
        return self.optimized_actions

    def animate_result(self, state, action):
        self.env.reset()
        self.set_state(state)
        for k in range(len(action)):
            # env.env.env.mujoco_render_frames = True
            self.env.render()
            self.env.step([action[k]])
            time.sleep(0.2)
        # env.env.env.mujoco_render_frames = False


def check_model(dyn_network):
    act = dyn_network.env.action_space.sample()
    dyn_network.env.reset()
    obs = np.array([-1., 0., 0.03])  # [cos(theta), sin(theta), theta_dot] corresponding to theta = pi
    dyn_network.env.env.state = [np.pi, 0.03]
    print('action taken:', act)
    print('starting_state:', obs)
    s_pred = dyn_network.dyn([obs[None, :], act[None, :]])
    s1, _, _, _ = dyn_network.env.step(act)
    print('Actual state:', s1)
    print('Predicted state:', s_pred)


if __name__ == '__main__':
    N = 50
    env_name = 'Pendulum-v0'
    actor_lr = 0.001
    minibatch_size = 50
    update_freq = 64
    buffer_size = 1e06

    learn_dyn = LearnModel(env_name=env_name, actor_lr=0.001, minibatch_size=50, update_freq=64, buffer_size=1e06)

    learn_dyn.dyn.load_weights('training/pend_fwd_dyn_model')

    # learn_dyn.dyn is the forward dynamics model which gives the next state given the current state and action
    check_model(learn_dyn)

    # learn_dyn.train(train_steps=N)
    # print('After training')
    # check_model(learn_dyn)

    n_rollouts, horizon = 50, 20
    mpc = MPPI(learn_dyn.dyn, env_name=env_name, K=n_rollouts, T=horizon, render=False)
    act = mpc.control(iter=30)

    x0 = np.load('data/initial_state.npy', allow_pickle=True)
    print('final')
    mpc.animate_result(x0, act)
    print('done')




