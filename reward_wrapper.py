import gymnasium as gym
from typing import SupportsFloat


class FrozenLakeRewardWrapper(gym.RewardWrapper):
    def __init__(
        self,
        env,
        goal_reward: SupportsFloat = 1,
        hole_reward: SupportsFloat = -1,
        step_reward: SupportsFloat = 0
    ):
        super().__init__(env)
        self.desc = self.env.unwrapped.desc

        self.G_reward = goal_reward
        self.H_reward = hole_reward
        self.F_reward = step_reward

    def reward(self, r: SupportsFloat) -> SupportsFloat:
        state = self.env.unwrapped.s  # State index

        # Decode current state byte [S, F, H, G] from env description
        state_desc = self.desc.flatten()[state].decode("utf-8").strip()

        if state_desc == 'G':
            # Reached the goal
            return self.G_reward
        elif state_desc == 'H':
            # Fell in a Hole
            return self.H_reward
        else:
            # All other steps
            return self.F_reward
