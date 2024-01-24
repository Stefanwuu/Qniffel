# Qniffel
A reinforcement learning based App to play the dice game Kniffel(Yahtzee)

## General information
The final goal of this project is to build an App that plays the game Kniffel(english: Yahtzee) automatically with super-human performance. More about the game can be found [here](https://en.wikipedia.org/wiki/Yahtzee).
The project contains two main parts: the training of a Kniffel agent using reinforcement learning and the implementation of an iOS App that plays Kniffel with physical dice based on the trained agent.
The training is based on open-ai gym and Pytorch. The iOS App will be implemented using SwiftUI.  

## RL agent training
The plan is to test two different reinforcement learning algorithms. One is the [deep Q-network](https://arxiv.org/abs/1312.5602)(DQN), 
which was applied [here](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf) for the game of Yahtzee. 
The other one is the [advantage actor-critic algorithm](https://arxiv.org/abs/1602.01783)(A2C), whose performance was reported [here](https://dionhaefner.github.io/2021/04/yahtzotron-learning-to-play-yahtzee-with-advantage-actor-critic/#pre-training-via-advantage-look-up-table) for the game of Yahtzee.
### Training details
- We only optimize the agent under single-player mode. Multi-player mode is although potentially helpful for training, it could be difficult for agent to learn the strategy(see discussion [here](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf)).
- Augmented input feature similar to [this paper](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf) is used. 
The 112-dimensional input feature encodes the current round of the game, the current dice roll, sum of dice, availability of score categories, and current upper scores(for bonus).
- Invalid action masking is used to prevent the agent from choosing invalid actions in the game(e.g. choosing already used score category).
### DQN model
- [Double DQN](https://arxiv.org/abs/1509.06461) is used to improve performance.
- NN structure: 2 linear hidden layers of 128 units with ReLU activation.
- Tricks for convergence:
  - lower target network update frequency( around 2000 steps) to stabilize training
  - linear decay of epsilon requires less tuning and works well
  - auxiliary feature to help agent understand the game(e.g. order of dice is irrelevant).

### Setup
The gym environment is mostly from [this repository](https://github.com/villebro/gym-yahtzee) with some modification of joker rules and debugging.
All the experimental results are available on [wandb](https://wandb.ai/naiv/Qniffel?workspace=user-naiv).
For simplicity, all training is done on a MacBook Pro with M1 Pro chip using the mps backend extension of pytorch.
### Performance

| Agent                                                                                                                                                             | avg score in 3000 games |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| random                                                                                                                                                            | 46.5                    |
| greedy                                                                                                                                                            | TODO                    | 
| DQN                                                                                                                                                               | 106.5                   |
| DQN in [paper](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf)                                                                                     | 77.8                    |
| A2C                                                                                                                                                               | TODO                    |
| A2C in [article](https://dionhaefner.github.io/2021/04/yahtzotron-learning-to-play-yahtzee-with-advantage-actor-critic/#pre-training-via-advantage-look-up-table) | 239.7                   |

## iOS App
### functionality
- Detection and understanding of dice roll as in [example](https://developer.apple.com/documentation/vision/understanding_a_dice_roll_with_vision_and_object_detection) using Vision framework
- Understandable instruction for user to roll the dice
- Recognition and handling the situation where the instruction is not followed
- Correct loading of Pytorch model + feature conversion
- Automatic calculation of score