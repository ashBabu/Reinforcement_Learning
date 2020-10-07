import numpy as np
import gym
import time
import tensorflow as tf
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


class MPPI:
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, nn_dyn_model=None, current_state=[np.pi, 0.15], env_name='Pendulum-v0', K=200, T=20, gamma=0.98, U=None, lambda_=1.0,
                 noise_mu=0, noise_sigma=1, u_init=1, render=False, downward_start=False, save=True):
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

    def reward(self, state, action):
        """ For pendulum, obs = [cos(theta), sin(theta), theta_dot]"""
        th, thdot = state
        # if tf.is_tensor(curr_obs):
        #     curr_obs = curr_obs.numpy()[0]
        # th = np.arctan2(curr_obs[1], curr_obs[0])
        reward = -(self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (action[0] ** 2))
        return reward

    def state2obs(self, state):
        return np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

    def obs2state(self, obs):
        return np.array([np.arctan2(obs[1], obs[0]), obs[2]])

    def _compute_total_cost(self, state, k):
        self.set_state(state)
        cost = 0
        obs = self.state2obs(state)
        for t in range(self.horizon):
            perturbed_actions = self.U[t] + self.noise[k, t]
            actions = np.clip(perturbed_actions, self.min_torque, self.max_torque)
            if not self.fwd_dyn:
               next_obs, reward, done, info = self.env.step(actions)
               rew = self.reward(state, actions)
               state = self.obs2state(next_obs)
               print(reward, rew, '##:', rew - reward)
            else:
                next_obs = self.fwd_dyn([obs[None, :], actions[None, :]])
                reward = self.reward(state, actions)
                obs = next_obs.numpy()[0]
                state = self.obs2state(obs)
            cost += self.gamma ** t * -reward
        return cost

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def control(self, iter=200):
        for zz in range(iter):
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
            act = np.array([self.U[0]])[None, :]
            act = np.clip(act, self.min_torque, self.max_torque)
            # print('state b4', self.get_state())
            if not self.fwd_dyn:
                s, r, _, _ = self.env.step(act[0])
            else:
                obs = self.state2obs(self.x_init)
                s = self.fwd_dyn([obs[None, :], act[0]])
                st = self.obs2state(s.numpy()[0])
                r = self.reward(st, act)
                self.set_state(st)
            # print('state after', self.get_state())
            self.optimized_actions.append(act[0][0])
            self.x_init = self.get_state()
            self.U = np.roll(self.U, -1, axis=0)  # shift all elements to the left
            self.U[-1] = 1.  #
            print("iter no: %0.2f action taken: %.2f cost received: %.2f" % (zz, act[0][0], -r))
            if self.render:
                self.env.render()
            self.cost_total[:] = 0
        return np.array(self.optimized_actions)

    def animate_result(self, state, action):
        self.env.reset()
        self.set_state(state)
        for k in range(len(action)):
            # env.env.env.mujoco_render_frames = True
            self.env.render()
            self.env.step([action[k]])
            time.sleep(0.2)
        # env.env.env.mujoco_render_frames = False


if __name__ == '__main__':
    N = 50
    env_name = 'Pendulum-v0'
    actor_lr = 0.001
    minibatch_size = 50
    update_freq = 64
    buffer_size = 1e06
    n_rollouts, horizon = 40, 40

    learn_dyn = LearnModel(env_name=env_name, lr=0.001, minibatch_size=50, update_freq=64, buffer_size=1e06)

    learn_dyn.dyn.load_weights('training/pend_fwd_dyn_model_64')

    # learn_dyn.dyn is the forward dynamics model which gives the next state given the current state and action
    learn_dyn.check_model()
    #
    # mpc = MPPI(learn_dyn.dyn, env_name=env_name, K=n_rollouts, T=horizon, render=False)
    # opt_act = mpc.control(iter=3)[0]


    # learn_dyn.train(train_steps=N)
    # print('After training')
    # check_model(learn_dyn)

    train, Dynamics = False, False

    def decision(is_train=False, action_save_name='opt_act', iter=100):
        if is_train:
            act = mpc.control(iter=iter)
            np.save('data/%s' % action_save_name, act, allow_pickle=True)
            x0 = np.load('data/initial_state.npy', allow_pickle=True)
            mpc.animate_result(x0, act)
        else:
            x0 = np.load('data/initial_state.npy', allow_pickle=True)
            opt_act = np.load('data/%s' % action_save_name, allow_pickle=True)
            print('final')
            mpc.animate_result(x0, opt_act)
            print('done')

    if Dynamics:
        ####  DYNAMICS IS KNOWN AND IS USED AS 'ENV.STEP(aCTION)' ##########
        mpc = MPPI(env_name=env_name, K=n_rollouts, T=horizon, render=False)
        ####################################################################
        decision(is_train=train, action_save_name='optimizedActions_knownDynamics.npy')
    else:
        #### UNKNOWN DYNAMICS AND HENCE A FUNCTION APPROXIMATOR IS PASSED ON TO MPPI #########
        mpc = MPPI(learn_dyn.dyn, env_name=env_name, K=n_rollouts, T=horizon, render=False)
        ######################################################################################
        decision(is_train=train, action_save_name='optimizedActions_unknownDynamics.npy')




