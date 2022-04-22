import gym
import numpy as np


class Env(gym.Env):
    def __init__(self, imb_ratio: float, X_train: np.ndarray, Y_train: np.ndarray, random=True):
        super().__init__()
        self.training_step = 1
        self.step_count = 0
        self.x = X_train
        self.y = Y_train
        self.random = random
        self.dataset_idx = 0
        self.id = np.arange(self.x.shape[0])  # List of IDs to connect X and y data
        self.imb_ratio = imb_ratio
        self._episode_ended = False

    def get_reward(self, action, env_action):
        if action == env_action:
            if env_action == 1:  # Minority
                reward = 1  # True Positive
            else:
                reward = self.imb_ratio  # True Negative
        else:
            if env_action == 1:  # Minority
                reward = -1  # False Positive
            else:  # Majority
                reward = -self.imb_ratio  # False Negative
        return reward

    def step(self, action):
        # if self._episode_ended:
        #     # The last action ended the episode. Ignore the current action and start a new episode
        #     return
        self._episode_ended = False
        env_action = self.y[self.id[self.step_count]]  # y
        # print('action', action)
        # print('env_action', env_action)

        reward = Env.get_reward(self, action, env_action)
        if reward == -1:
            self._episode_ended = True  # Stop episode when minority class is misclassified

        # if self.step_count == self.x.shape[0] - 2:
        # if self.step_count == self.training_step:
        #     self._episode_ended = True

        self.step_count += 1
        next_obs = self.x[self.id[self.step_count]]

        return next_obs, reward, self._episode_ended

    def start(self):
        self.step_count = 0  # Reset episode step counter at the begin of training
        self.obs = self.x[self.id[self.step_count]]
        self._episode_ended = False  # Reset terminal condition
        return self.obs

    # def calculate_Q_value(self, action_space, env_action):
    #     Q = []
    #     for action in range(action_space):
    #         reward = Env.get_reward(self, action, env_action)
    #         Q.append(reward)
    #     return Q
