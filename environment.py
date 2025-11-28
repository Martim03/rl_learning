from typing import SupportsFloat
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from reward_wrapper import FrozenLakeRewardWrapper
# TODO - maybe use only valid actions
# ? TODO - change this class to inheritence?
# TODO - make averega/max reward graphs


class FrozenLakeEnviroment:
    def __init__(
        self,
        goal_reward: SupportsFloat,
        hole_reward: SupportsFloat,
        step_reward: SupportsFloat,
        max_steps: int = 50,
    ):
        base_env = gym.make(
            "FrozenLake-v1", render_mode="human", is_slippery=False)
        # Actions: (0: Left, 1: Down, 2: Right, 3: Up)
        # States: 0-15 (4x4 grid)

        ###### WRAPPERS ######
        time_limit_env = TimeLimit(  # MAX STEPS WRAPPER
            base_env, max_episode_steps=max_steps)

        self.env = FrozenLakeRewardWrapper(  # CUSTOM REWARDS WRAPPER
            env=time_limit_env,
            goal_reward=goal_reward,
            hole_reward=hole_reward,
            step_reward=step_reward
        )

        ######################

        self.unwrapped = self.env.unwrapped

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
