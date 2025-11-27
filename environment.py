import gymnasium as gym
# TODO - make a wrapper
# TODO - maybe use only valid actions
# TODO - change reward system
# TODO - change truncation to 50 steps


class FrozenLakeEnviroment:
    def __init__(self):
        self.env = gym.make(
            "FrozenLake-v1", render_mode="human", is_slippery=False)
        # Actions: (0: Left, 1: Down, 2: Right, 3: Up)
        # States: 0-15 (4x4 grid)

    def get_action_label(self, action: int) -> str:
        action_dic = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
        return action_dic.get(action)

    def reset(self):
        """Delegates reset to the underlying environment."""
        return self.env.reset()

    def step(self, action):
        """Delegates step to the underlying environment."""
        # Returns: (new_state, reward, terminated, truncated, info)
        return self.env.step(action)

    def close(self):
        """Delegates close to the underlying environment."""
        return self.env.close()

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space
