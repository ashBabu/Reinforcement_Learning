import gym
import jax.numpy as jnp
import pickle
import numpy as np
import time
from jax import grad, jacobian, jacfwd
# try:
#     #note autograd should be replacable by jax in future
#     # import autograd.numpy as np
#     import jax.numpy as np
#     from jax import grad, jacobian, jacfwd
#     has_autograd = True
# except ImportError:
#     import numpy as np
#     has_autograd = False

# import jax.numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as opt
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization
from tensorflow.keras import Model
import random
from collections import deque

print('This will work only with Tensorflow 2.x')


class ReplayBuffer:

    def __init__(self, buffer_size, save=True):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.save = save
        self.state_experinece = {'obs': [], 'actions': [], 'rewards': [],
                            'dones': [], 'next_obs': []}

    def add(self, s, a, r, d, s2):
        if self.save:
            self.state_experinece['obs'].append(s)
            self.state_experinece['actions'].append(a)
            self.state_experinece['rewards'].append(r)
            self.state_experinece['dones'].append(d)
            self.state_experinece['next_obs'].append(s2)
        np.save('data/state_experience.npy', self.state_experinece, allow_pickle=True)
        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        r_num = random.randrange(0, int(self.count/3))
        if self.count < batch_size:
            batch = self.buffer
        else:
            # if r_num + batch_size > self.count:
            #     batch = [self.buffer[index] for index in range(r_num, self.count)]
            # else:
            #     batch = [self.buffer[index] for index in range(r_num, batch_size+r_num)]
            batch = [self.buffer[index] for index in range(self.count-batch_size, self.count)]

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        d_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def load_(self, name='state_experience'):
        data = np.load('data/%s' % name, allow_pickle=True)
        # assert(data, dict)
        obs = np.array(data.item().get('obs'))
        actions = np.array(data.item().get('actions'))
        rewards = np.array(data.item().get('rewards'))
        dones = np.array(data.item().get('dones'))
        next_obs = np.array(data.item().get('next_obs'))
        return obs, actions, rewards, dones, next_obs


class FwdDynamics:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input')
        state_out = Dense(40, activation='relu')(state)
        state_out = BatchNormalization()(state_out)
        state_out = Dense(32, activation="relu")(state_out)
        state_out = BatchNormalization()(state_out)

        action = Input(shape=(self.action_dim,), name='action_input')
        action_out = Dense(32, activation="relu")(action)
        action_out = BatchNormalization()(action_out)

        x = concatenate([state_out, action_out])
        out = BatchNormalization()(x)
        out = Dense(64, activation="relu")(out)
        out = BatchNormalization()(out)
        out = Dense(self.state_dim, activation='linear')(out)
        return Model(inputs=[state, action], outputs=out)


class LearnModel:
    def __init__(self, env_name='Pendulum-v0', lr=0.001, minibatch_size=50, update_freq=64, buffer_size=1e06):
        self.env = gym.make(env_name)
        self.dt = self.env.dt
        self.a_dim = self.env.action_space.shape[0]
        self.s_dim = self.env.observation_space.shape[0]
        self.lr = lr
        self.dyn = FwdDynamics(self.s_dim, self.a_dim).model()
        self.dyn_opt = opt.Adam(learning_rate=self.lr)

        self.update_freq = update_freq
        self.minibatch_size = minibatch_size
        self.buffer_size = buffer_size  # 000
        self.buffer = ReplayBuffer(self.buffer_size)

    """
    def dynamics(x, u):
        env.reset()
        env.env.state = x
        s_next, _, _, _ = env.step(u)
        return s_next
    
    # jax.device_put(x)[idx]
    
    aa = lambda x: dynamics(x, u)
    bb = lambda u: dynamics(x, u)
    
    x, u = [jnp.pi, 0.21], [1]
    A = jacfwd(aa)(x)
    B = jacfwd(bb)(u)
    """

    def policy(self, observation):  # random policy
        return self.env.action_space.sample()
        # return np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape)

    def do_update(self):
        """
        Trying to find the increment in states, f_(theta), from the equation
        s_{t+1} = s_t + dt * f_(theta)(s_t, a_t)
        """
        s_batch, a_batch, r_batch, d_batch, s2_batch = self.buffer.sample_batch(self.minibatch_size)
        obs_actual_error = (s2_batch - s_batch)/self.dt

        with tf.GradientTape() as tape:
            obs_pred_error = self.dyn([s_batch, a_batch])
            loss = tf.math.reduce_mean(tf.math.square(obs_actual_error - obs_pred_error))
        dyn_grad = tape.gradient(loss, self.dyn.trainable_variables)
        self.dyn_opt.apply_gradients(zip(dyn_grad, self.dyn.trainable_variables))

    def train(self, train_steps=1000000):
        x0 = self.env.reset()
        print('Model evaluation started')
        for i in range(train_steps):
            a = self.policy(x0)
            s_, r, done, _ = self.env.step(a)
            self.buffer.add(np.reshape(x0, (self.s_dim,)), np.reshape(a, (self.a_dim,)), r,
                                   done, np.reshape(s_, (self.s_dim,)))
            if not self.buffer.size() % self.update_freq:
                self.do_update()
            x0 = s_

    def save_weights(self, nn_network, save_name='final_weights'):
        nn_network.save_weights("training/%s.h5" % save_name)
        # to save in other format
        nn_network.save_weights('training/%s' % save_name, save_format='tf')
        print('Training completed and network weights saved')

    def load_weights(self, nn_network, name='pend_fwd_dyn_model'):
        network = nn_network(self.s_dim, self.a_dim).model()
        network.load_weights('training/%s' % name)
        return network

    def check_model(self):
        s0 = self.env.reset()
        print('starting_state:', s0)
        s0 = s0[None, :]
        for _ in range(100):
            act = self.env.action_space.sample()
            obs_actual, _, _, _ = self.env.step(act)
            obs_pred = s0 + self.dt * self.dyn([s0, act[None, :]])
            s0 = obs_pred

        # obs = np.array([-1., 0., 0.03])  # [cos(theta), sin(theta), theta_dot] corresponding to theta = pi
        # self.env.env.state = [np.pi, 0.03]
        # print('action taken:', act)
        print('current_state:', obs_actual)
        print('Predicted state:', obs_pred)


if __name__ == '__main__':
    saveWeights = True
    modellearn = LearnModel(env_name='Pendulum-v0', lr=0.001, minibatch_size=64, update_freq=50, buffer_size=1000)
    iterations = 1000000
    if saveWeights:
        """pend_fwd_dyn_model is a fully trained fwd dynamic model. It has 512 neurons in the line
        out = Dense(512, activation="relu")(out) """
        now = time.time()
        modellearn.train(iterations)
        print('Time taken for %.2f iterations is' %iterations, time.time()-now)
        modellearn.save_weights(modellearn.dyn, save_name='pend_fwd_dyn_model_obs_error')
    else:
        modellearn.dyn.load_weights('training/pend_fwd_dyn_model_64')

    theta = np.linspace(0, 2*np.pi, 100)
    # act = modellearn.env.action_space.sample()
    act = np.array([2.])
    modellearn.check_model()

    print('done')


