from q_table import Q_table
from environment import FrozenLakeEnviroment
from time import sleep


class FrozenLakeAgent:
    def __init__(self):
        self.env = FrozenLakeEnviroment()
        self.q_table = Q_table(
            learning_rate=0.001,
            epsilon_max=0.1,
            epsilon_decay=0.01,
            epsilon_min=0.01,
            discount_factor=0.95
        )

    def get_action(self):
        return self.q_table.get_action()

    def run_episode(self):
        state, _ = self.env.reset()
        step = 0
        total_reward = 0
        
        while True:
            action = self.get_action()  # Random action

            new_state, reward, terminated, truncated, _ = self.env.step(action)
            print(f"Step {step}: state={state} -> new_state={new_state}")
            print(
                f"\taction={self.env.get_action_label(action)}, reward={reward}")

            state = new_state
            total_reward += reward
            step += 1

            if truncated or terminated:
                # Terminated -> game ended
                # Truncated -> step limit reached
                break

            sleep()  # Wait a bit befor next action

        self.env.close
        return total_reward

    def train(self):
        print("Starting")
        accumulated_reward = self.run_episode()
        print(f"Episode ended with total reward: {accumulated_reward}")
