"""Run Lunar Lander DDQN training (short demo by default)."""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import gymnasium as gym
import numpy as np
import time
from ddqn_torch import DoubleQAgent

LEARN_EVERY = 4

def train_agent(n_episodes=2000, load_latest_model=False):
    print("Training a DDQN agent on {} episodes. Pretrained model = {}".format(n_episodes, load_latest_model))
    env = gym.make("LunarLander-v3")
    agent = DoubleQAgent(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)
    if load_latest_model:
        agent.load_saved_model("ddqn_torch_model.h5")
        print("Loaded most recent: ddqn_torch_model.h5")

    scores = []
    start = time.time()
    for i in range(n_episodes):
        terminated = False
        truncated = False
        score = 0
        state = env.reset()[0]
        steps = 0
        while not (terminated or truncated):
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            agent.save(state, action, reward, new_state, terminated)
            state = new_state
            if steps > 0 and steps % LEARN_EVERY == 0:
                agent.learn()
            steps += 1
            score += reward

        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100) : (i + 1)])

        if (i + 1) % 10 == 0 and i > 0:
            print(
                "Episode {} in {:.2f} min. [{:.2f}/{:.2f}]".format(
                    (i + 1), (time.time() - start) / 60, score, avg_score
                )
            )

        if (i + 1) % 100 == 0 and i > 0:
            agent.save_model("ddqn_torch_model.h5")

    env.close()
    return agent

def watch_agent(model_path="ddqn_torch_model.h5", n_episodes=3):
    """Load trained model and run with on-screen rendering."""
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = DoubleQAgent(gamma=0.99, epsilon=0.0, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)
    agent.load_saved_model(model_path)
    print("Watching agent for {} episodes. Close the window to stop.".format(n_episodes))
    for ep in range(n_episodes):
        state, _ = env.reset()
        score = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
        print("  Episode {} score: {:.1f}".format(ep + 1, score))
    env.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes (default 20 for quick demo)")
    p.add_argument("--load", action="store_true", help="Load existing model from ddqn_torch_model.h5")
    p.add_argument("--watch", action="store_true", help="Watch trained agent (loads ddqn_torch_model.h5)")
    p.add_argument("--model", type=str, default="ddqn_torch_model.h5", help="Model path for --watch")
    args = p.parse_args()
    if args.watch:
        watch_agent(model_path=args.model)
    else:
        train_agent(n_episodes=args.episodes, load_latest_model=args.load)
