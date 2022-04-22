import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tf_agents.utils import common
from classified_env_dev import Env

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20


class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = Sequential()
        self.model.add(Dense(300, activation="relu", input_shape=(observation_space,)))
        self.model.add(Dense(150, activation="relu"))
        self.model.add(Dense(50, activation="relu"))
        self.model.add(Dense(2, activation="softmax"))
        self.model.compile(loss=common.element_wise_squared_loss, optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, observation, action, reward, done):
        # print("Action: " + str(action))
        # print("Done: " + str(done))
        # print("Reward: " + str(reward))
        if not done:
            q_action = reward + self.model.predict(observation)[0][action]
            if q_action > 1:
                q_action = 1
            if q_action < 0:
                q_action = 0
            q_not_action = 1 - q_action
        else:
            q_action = 0
            q_not_action = 1

        q_values = self.model.predict(observation)
        # print(q_values)
        q_values[0][action] = q_action
        q_values[0][1 - action] = q_not_action
        # print(q_values)
        # print("===========")
        self.memory.append((observation, q_values))

    def act(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self, batch):
        obs = []
        q_update = []
        for observation, q_value in self.memory:
            obs.append(observation.ravel())
            q_update.append(q_value.ravel())
        obs = np.reshape(obs, [len(self.memory), 6])
        q_update = np.reshape(q_update, [len(self.memory), 2])
        self.model.fit(obs, q_update, batch_size=batch, verbose=1)
