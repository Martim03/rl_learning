from random import sample

class Q_table:
    def __init__(
        self,
        learning_rate: float,
        epsilon_max: float,
        epsilon_decay: float,
        epsilon_min: float,
        discount_factor: float
    ):
        self.lr = learning_rate
        self.eps = epsilon_max
        self.discount = discount_factor

    def decay_epsilon(self):
        # ! TODO - implement
        pass

    def get_action(self):
        # ! TODO - implement
        return sample([0, 1, 2, 3], 1)[0]

    def update_table(self):
        # ! TODO - implement
        pass
