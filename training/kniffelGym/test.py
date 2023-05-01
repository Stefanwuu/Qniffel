from training.kniffelGym.envs import KniffelSingleEnv
import gym

env: KniffelSingleEnv = gym.make('kniffel-single-v0')
env.reset()
for t in range(1000):
    env.render()
    action = env.sample_action()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break
