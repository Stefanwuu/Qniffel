import wandb

class agent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def train(self):
        pass

    def decide(self, state):
        return -1

    def evaluate(self):
        cum_reward = 0
        cur_reward = 0
        max_reward = 0
        for _ in range(3000):   # 3000 games simulated
            self.env.reset()
            state = self.env.get_observation_space()
            for _ in range(100):    # only 36 sub rounds in total
                action = self.decide(state)
                state, reward, done, _, _ = self.env.step(action)
                cum_reward += reward
                cur_reward += reward
                if done:
                    if max_reward < cur_reward:
                        max_reward = cur_reward
                    cur_reward = 0
                    break

        wandb.log({'reward': cum_reward / 3000, 'max_reward': max_reward})