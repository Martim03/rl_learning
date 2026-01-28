from q_table import Q_table
from environment import FrozenLakeEnviroment
from time import sleep


class FrozenLakeAgent:
    def __init__(self):
        # *CHANGE PARAMETERS HERE*
        self.env = FrozenLakeEnviroment(
            max_steps=80,
            goal_reward=5,
            hole_reward=-1,
            step_reward=-0.01
        )

        self.q_table = Q_table(
            observation_space_n=self.env.observation_space().n,
            action_space_n=self.env.action_space().n,
            learning_rate=0.1,
            epsilon_max=1,
            epsilon_decay=0.01,  # 0.01 -> 100 episodes = 0.37
            epsilon_min=0.01,
            discount_factor=0.99
        )
        
        self.max_episodes = 5000

    def get_action(self, state: int) -> int:
        return self.q_table.get_action(state)

    def step_update(self, state: int, action: int, next_state: int, reward: int):
        self.q_table.update_table(state, action, next_state, reward)

    def episode_update(self, episode: int):
        self.q_table.decay_epsilon(episode)

    def run_episode(self) -> tuple[int, int]:
        state, _ = self.env.reset()
        step = 1
        total_reward = 0

        while True:
            # Select Action
            action = self.get_action(state)

            # Get new observation and reward
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            print(f"Step {step}: state={state} -> new_state={new_state}")
            print(
                f"\taction={self.env.get_action_label(action)}, reward={reward}")

            # Update Q-value
            self.step_update(state, action, new_state, reward)
            state = new_state
            total_reward += reward
            step += 1

            # Check termination
            if truncated or terminated:
                """ 
                Terminated -> game ended
                Truncated -> step limit reached 
                """
                break

            # Wait a bit before next step
            sleep(0)

        self.env.close
        return step, total_reward

    def train(self):
        print("Starting...")

        for episode in range(1, self.max_episodes+1):
            print (f"\nEpisode {episode} now starting...")
            
            print("\n=======================")
            steps, accumulated_reward = self.run_episode()
            print("=======================\n")

            self.episode_update(episode)
            print(
                f"> Episode {episode} ended with {steps} steps and total reward: {accumulated_reward}")
            episode += 1
