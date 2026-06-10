from Dualenv import Dualenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Monitor import Monitor
import matplotlib.pyplot as plt

################################################# Define Variables ########################################################################

log_dir = "log"

# Best model saved during training
modelName = log_dir + "/best.zip"

# VecNormalize file saved at the end of train.py
envName = log_dir + "/inSiAd2.pkl"

################################################# Testing and Evaluation #################################################################

env = Dualenv(renders=True, is_discrete=False, max_steps=1024)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])

# Load normalization stats from training
env = VecNormalize.load(envName, env)
env.training = False
env.norm_reward = False

# Load trained model
model = PPO.load(modelName, env=env)

test = 100
success_count = 0

success_rates = []
episode_numbers = []

for i in range(test):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]

    # success gives a +100000 reward bonus in Dualenv.py
    success = episode_reward >= 100000
    if success:
        success_count += 1

    current_success_rate = (success_count / (i + 1)) * 100
    success_rates.append(current_success_rate)
    episode_numbers.append(i + 1)

    print(f"Episode {i + 1}: reward={episode_reward}, success={success}, success rate={current_success_rate:.2f}%")

print("SUCCESS RATE IS:", str((success_count / test) * 100) + "%")
print("Evaluation is Done")


"""
Success Rates Graph
"""
plt.figure(figsize=(8, 5))
plt.plot(episode_numbers, success_rates, marker='o')
plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate Over Evaluation Episodes")
plt.ylim(0, 100)
plt.grid(True)
plt.show()

env.close()












