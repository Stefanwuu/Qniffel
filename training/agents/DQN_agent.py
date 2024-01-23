import wandb
from training.agents.agent import agent

from collections import namedtuple, deque
import random
import math
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# -- NN structure of DQN -- #
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, config):
        super(DQN, self).__init__()
        self.input = nn.Linear(n_observations, config.get('hidden_size', 256))
        self.hidden = nn.ModuleList()
        for _ in range(config.get('num_hidden_layers', 1)):
            self.hidden.append(nn.Linear(config.get('hidden_size', 256), config.get('hidden_size', 256)))
        self.output = nn.Linear(config.get('hidden_size', 256), n_actions)

    def forward(self, x):
        x = F.relu(self.input(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output(x)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# -- replay buffer to store state-action transitions -- #
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(agent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # apple silicon

        # wandb to monitor training
        wandb.init(
            project="Qniffel",
            group="DQN",
            name=config['name'],
            config=config  # track hyperparameters and run metadata
        )

        # -- initialize policy and target network -- #
        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        state = self.transform_state(state)
        self.n_observations = state.size(-1)
        print(f'Observation space: {self.n_observations}, Action space: {self.n_actions}')

        self.policy_net = DQN(self.n_observations, self.n_actions, config).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config['lr'], amsgrad=True)
        self.memory = ReplayMemory(config['memory_size'])

        self.steps_done = 0

    # -- convert state to input feature -- #
    @staticmethod
    def get_one_hot(value, min_value, max_value):
        one_hot_list = []
        for i in range(min_value, max_value + 1):
            one_hot_list.append(1 if value == i else 0)
        return one_hot_list

    def transform_state(self, state):
        # convert gym observation to one-hot feature tensor([1, 112])
        final_list = []
        final_list += self.get_one_hot(state[0], 0, 12)  # round
        final_list += self.get_one_hot(state[1], 0, 2)  # sub_round
        final_list += self.get_one_hot(state[2], 1, 6)  # die 1
        final_list += self.get_one_hot(state[3], 1, 6)  # die 2
        final_list += self.get_one_hot(state[4], 1, 6)  # die 3
        final_list += self.get_one_hot(state[5], 1, 6)  # die 4
        final_list += self.get_one_hot(state[6], 1, 6)  # die 5

        # append auxiliary dice features
        # use group to indicate dice sum
        dice_sum = sum(state[2:7])
        final_list += self.get_one_hot(dice_sum // 4, 0, 7)
        # number of dice with value _
        for i in range(1, 7):
            final_list += self.get_one_hot(state[2:7].count(i), 0, 5)

        # append score as feature
        upper_score = 0
        for category_index in range(7, 22):
            category_score = state[category_index]
            if category_index <= 12:
                upper_score += max(0, category_score)
            # 0 for available
            final_list.append(0 if category_score < 0 else 1)
        # use group to indicate upper score
        upper_score_cat = upper_score // 10 if upper_score <= 63 else 6
        final_list += self.get_one_hot(upper_score_cat, 0, 6)

        return torch.tensor([final_list], dtype=torch.float, device=self.device)

    # -- invalid action masking -- #
    def mask_invalid_actions(self, policy):
        valid_actions = self.env.kniffel.get_possible_actions()
        assert valid_actions, 'No valid actions'
        for j in range(len(policy)):
            if j not in valid_actions:
                policy[j] = -float('inf')
        return policy

    # Epsilon greedy policy
    def select_action(self, state):
        sample = random.random()
        # eps linear decay
        total_steps = self.config['num_episodes'] * 36
        slope = (self.config['eps_start'] - self.config['eps_end']) / total_steps
        eps_threshold = max(self.config['eps_end'], self.config['eps_start'] - self.steps_done * slope)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # remove invalid actions by setting their policy to -inf
                policy = self.policy_net(state).squeeze(0)  # squeeze batch dim
                policy = self.mask_invalid_actions(policy)
                # t.max(0): tuple (value, index) of largest elem along feature axis(0)
                return policy.max(0)[1].view(1, 1)
        else:
            # exploration
            # only sample from possible actions(sample_action)
            action = self.env.sample_action()
            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    # -- for evaluation -- #
    def decide(self, state):
        state = self.transform_state(state)
        with torch.no_grad():
            policy = self.policy_net(state).squeeze(0)  # squeeze batch dim
            policy = self.mask_invalid_actions(policy)
        # t.max(0): tuple (value, index) of largest elem along feature axis(0)
        return policy.max(0)[1].view(1, 1).item()

    def optimize_model(self):
        if len(self.memory) < self.config['batch_size']:
            # fill the replay buffer first
            return
        transitions = self.memory.sample(self.config['batch_size'])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.config['batch_size'], device=self.device)
        if self.config.get('double_dqn', False):
            # Double DQN
            # Use policy_net to select the action, and use target_net to evaluate the action
            with torch.no_grad():
                next_state_action = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states). \
                    gather(1, next_state_action).squeeze(1)
        else:
            # DQN
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.config['gamma']) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        wandb.log({'loss': loss.item()})

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping: avoid nan
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # -- training loop -- #
    def train(self):
        num_episodes = self.config['num_episodes']

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            self.env.reset()
            state = self.env.get_observation_space()
            state = self.transform_state(state)
            cum_reward = 0
            for t in count():  # fixed episode len: impossible to be clipped
                action = self.select_action(state)
                observation, reward, terminated, _, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                cum_reward += reward.item()

                if terminated:
                    next_state = None
                else:
                    next_state = self.transform_state(observation)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                if self.config.get('target_update', None):
                    # Hard update of the target network's weights
                    if self.steps_done % self.config['target_update'] == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                else:
                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = (policy_net_state_dict[key] * self.config['tau'] +
                                                      target_net_state_dict[key] * (1 - self.config['tau']))
                    self.target_net.load_state_dict(target_net_state_dict)

                if terminated:
                    wandb.log({'cum_reward': cum_reward})
                    break

            if (i_episode + 1) % 1000 == 0 or (i_episode + 1) in self.config['test_episodes']:
                self.evaluate()
                torch.save(self.policy_net.state_dict(), f'./models/{self.config["name"]}_{i_episode+1}.pt')
        wandb.finish()