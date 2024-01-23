from training.kniffelGym.envs import KniffelSingleEnv
import gym

env = gym.make('kniffel-single-v0', render_mode='human')
env.reset()
for t in range(1000):
    env.render()
    action = env.sample_action()
    observation, reward, done, truncated, info = env.step(action)  # truncated is not used
    print(info['action_description'])
    if done or truncated:
        print("Episode finished after {} timesteps".format(t + 1))
        break
