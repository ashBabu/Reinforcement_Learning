import numpy as np
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization
from tensorflow.keras import Model


class Actor:
    def __init__(self, state_dim, action_dim, goal_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input')
        state_out = Dense(40, activation='relu')(state)
        state_out = BatchNormalization()(state_out)
        state_out = Dense(32, activation="relu")(state_out)
        state_out = BatchNormalization()(state_out)

        goal = Input(shape=(self.goal_dim,), name='goal_input')
        goal_out = Dense(32, activation="relu")(goal)
        goal_out = BatchNormalization()(goal_out)

        x = concatenate([state_out, goal_out])
        out = BatchNormalization()(x)
        out = Dense(64, activation="relu")(out)
        out = BatchNormalization()(out)
        out = Dense(self.action_dim, activation='linear')(out)
        return Model(inputs=[state, goal], outputs=out)


class Critic:
    def __init__(self, state_dim, action_dim, goal_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

    def model(self):
        # state = Input(shape=self.state_dim, name='state_input', dtype='float32')
        state = Input(shape=self.state_dim, name='state_input')
        state_out = Dense(40, activation='relu')(state)
        state_out = BatchNormalization()(state_out)
        state_out = Dense(32, activation="relu")(state_out)
        state_out = BatchNormalization()(state_out)

        action = Input(shape=(self.action_dim,), name='action_input')
        action_out = Dense(32, activation="relu")(action)
        action_out = BatchNormalization()(action_out)

        goal = Input(shape=(self.goal_dim,), name='goal_input')
        goal_out = Dense(32, activation="relu")(goal)
        goal_out = BatchNormalization()(goal_out)

        x = concatenate([state_out, goal_out])
        out = BatchNormalization()(x)
        out = Dense(64, activation="relu")(out)
        out = BatchNormalization()(out)
        out = Dense(1, activation='linear')(out)
        return Model(inputs=[state, goal], outputs=out)
