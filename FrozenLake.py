import gymnasium as gym
import time

env = gym.make("FrozenLake-v1", render_mode="human")
# Actions: (0: Left, 1: Down, 2: Right, 3: Up)
# States: 0-15 (4x4 grid)

print("Starting game...")
state, _ = env.reset()

terminated = False
truncated = False  # time limit flag

# Game Loop
while not terminated and not truncated:
    # Take action
    random_action = env.action_space.sample()
    print(f"Current state: {state}, Taking action: {random_action}")

    # Get results
    new_state, reward, terminated, truncated, info = env.step(random_action)
    print(f"  -> New state: {new_state}, Reward: {reward}")

    state = new_state
    time.sleep(0.5)

if reward == 1.0:
    print("\nGame Won!")
else:
    print("\nGame Over!")
env.close()
