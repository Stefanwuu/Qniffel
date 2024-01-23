import training.kniffelGym
import gym
import training.agents.DQN_agent as dqn
import training.agents.random_agent as random
import copy

env = gym.make('kniffel-single-v0', render_mode='human')

# random agent: average reward 46.5
#random = random.random(env, {})
#random.evaluate()

config = {'name': 'baseline', 'hidden_size': 128, 'num_hidden_layers': 2, 'batch_size': 128, 'gamma': 0.99,
          'eps_start': 0.9, 'eps_end': 0.05, 'tau': 0.005, 'lr': 1e-4, 'num_episodes': 5000, 'memory_size': 10000,
          'target_update': 800, 'test_episodes': [4400, 4600, 4800]}

# base DQN: average reward 71.4
#baseline = dqn.DQNAgent(env, config)
#baseline.train()

# double DQN: 84.2
config_double = copy.deepcopy(config)
config_double['double_dqn'] = True
config_double['name'] = 'ddqn_baseline'
double = dqn.DQNAgent(env, config_double)
double.train()

# num episodes tuning: 90.8
config_long = copy.deepcopy(config)
config_long['num_episodes'] = 10000
config_long['name'] = '10000ep'
long = dqn.DQNAgent(env, config_long)
long.train()

# target update tuning
# 200 not converged
config_target = copy.deepcopy(config)
for update in [200, 500, 1100]:
    config_target['target_update'] = update
    config_target['name'] = 'target_update_' + str(update)
    target = dqn.DQNAgent(env, config_target)
    target.train()

# lr tuning
config_lr = copy.deepcopy(config)
for lr in [3e-4, 8e-5, 5e-5]:
    config_lr['lr'] = lr
    config_lr['name'] = 'lr_' + str(lr)
    lr = dqn.DQNAgent(env, config_lr)
    lr.train()

