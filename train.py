import gymnasium as gym
from stable_baselines3 import PPO
from main import RobloxCarEnv

def train():
    env = RobloxCarEnv()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log="./ppo_jeep_tensorboard",
        device="cpu"
    )

    print("\n=== TRAINER READY ===")
    print("→ Go to Roblox Studio and press PLAY now")
    print("→ You should see 'First observation received!' in this window soon\n")

    model.learn(total_timesteps=100000, log_interval=10)

    model.save("ppo_jeep_final")
    print("Training finished!")

if __name__ == "__main__":
    train()