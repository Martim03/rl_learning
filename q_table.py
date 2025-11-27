import random as rand
import numpy as np


class Q_table:
    def __init__(
        self,
        observation_space_n,
        action_space_n,
        learning_rate: float,
        epsilon_max: float,
        epsilon_decay: float,
        epsilon_min: float,
        discount_factor: float
    ):
        self.lr = learning_rate
        self.eps = epsilon_max
        self.eps_max = epsilon_max
        self.eps_min = epsilon_min
        self.eps_decay_rate = epsilon_decay
        self.discount = discount_factor
        self.actions_n = action_space_n
        self.q_values = np.zeros((observation_space_n, action_space_n))

    def decay_epsilon(self, episode: int):
        """
        Exponential decay -> decay exploration rate after each episode
        """
        self.eps = self.eps_min + \
            (self.eps_max - self.eps_min) * \
            np.exp(-self.eps_decay_rate * episode)
        print(f"Epsilon decayed to: {self.eps}")

    def get_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection strategy
            for values <= epsilon select random action (exploration)
            for values > epsilon select best action (exploitation)
        """
        if rand.uniform(0, 1) > self.eps:
            return np.argmax(self.q_values[state])
        else:
            return rand.randint(0, self.actions_n-1)

    def update_table(self, state: int, action: int, next_state: int, reward: int):
        # Q-value update formula
        current_q = self.q_values[state][action]
        target = reward + self.discount * np.max(self.q_values[next_state])
        error = target - current_q
        new_q = current_q + self.lr * error

        self.q_values[state][action] = new_q
