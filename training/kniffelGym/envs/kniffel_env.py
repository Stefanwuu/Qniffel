from collections import defaultdict
from enum import Enum
import logging
import sys
from typing import Dict, Optional, Sequence

from gym import Env, spaces

import numpy as np

from training.kniffelEngine import Kniffel, Category


log = logging.getLogger(__name__)


class GameType(Enum):
    SUDDEN_DEATH = 0,
    RETRY_ON_WRONG_ACTION = 1


def get_score(score: Optional[int]) -> int:
    return score if score is not None else -1


class KniffelSingleEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 game_type: GameType = GameType.RETRY_ON_WRONG_ACTION,
                 seed=None):
        self.kniffel = Kniffel(seed=seed)
        self.game_type = game_type
        self.action_space = spaces.Discrete(44)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(13),  # round
            spaces.Discrete(4),  # sub-round
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 1
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 2
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 3
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 4
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 5
            spaces.Box(low=-1, high=5, shape=(1,), dtype=np.int16),  # aces
            spaces.Box(low=-1, high=10, shape=(1,), dtype=np.int16),  # twos
            spaces.Box(low=-1, high=15, shape=(1,), dtype=np.int16),  # threes
            spaces.Box(low=-1, high=20, shape=(1,), dtype=np.int16),  # fours
            spaces.Box(low=-1, high=25, shape=(1,), dtype=np.int16),  # fives
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # sixes
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # three of a kind
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # four of a kind
            spaces.Box(low=-1, high=25, shape=(1,), dtype=np.int16),  # full house
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # small straight
            spaces.Box(low=-1, high=40, shape=(1,), dtype=np.int16),  # large straight
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # chance
            spaces.Box(low=-1, high=50, shape=(1,), dtype=np.int16),  # kniffel
            spaces.Box(low=-1, high=35, shape=(1,), dtype=np.int16),  # upper bonus
            spaces.Box(low=-1, high=1200, shape=(1,), dtype=np.int16),  # kniffel bonus
        ))

    def get_observation_space(self):
        kniffel = self.kniffel
        return (
            kniffel.round,
            kniffel.sub_round,
            kniffel.dice[0],
            kniffel.dice[1],
            kniffel.dice[2],
            kniffel.dice[3],
            kniffel.dice[4],
            get_score(kniffel.scores.get(Category.ACES)),
            get_score(kniffel.scores.get(Category.TWOS)),
            get_score(kniffel.scores.get(Category.THREES)),
            get_score(kniffel.scores.get(Category.FOURS)),
            get_score(kniffel.scores.get(Category.FIVES)),
            get_score(kniffel.scores.get(Category.SIXES)),
            get_score(kniffel.scores.get(Category.THREE_OF_A_KIND)),
            get_score(kniffel.scores.get(Category.FOUR_OF_A_KIND)),
            get_score(kniffel.scores.get(Category.FULL_HOUSE)),
            get_score(kniffel.scores.get(Category.SMALL_STRAIGHT)),
            get_score(kniffel.scores.get(Category.LARGE_STRAIGHT)),
            get_score(kniffel.scores.get(Category.CHANCE)),
            get_score(kniffel.scores.get(Category.KNIFFEL)),
            get_score(kniffel.scores.get(Category.UPPER_SECTION_BONUS)),
            get_score(kniffel.scores.get(Category.KNIFFEL_BONUS)),
        )

    def sample_action(self):
        action = self.kniffel.sample_action()
        log.info(f'Sampled action: {action}')
        return action

    def step(self, action: int):
        kniffel = self.kniffel
        try:
            reward = kniffel.take_action(action)
            finished = kniffel.is_finished()
            valid_move = True
        except Exception:
            valid_move = False
            reward = 0
            if self.game_type == GameType.SUDDEN_DEATH:
                log.info('Invalid action, terminating round.')
                finished = True
            else:  # retry on wrong action
                log.info('Invalid action, step ignored.')
                finished = False

        log.info(f'Finished step. Reward: {reward}, Finished: {finished}')
        debug_info = {
            'valid_move': valid_move,
        }
        return self.get_observation_space(), reward, finished, debug_info

    def reset(self, seed=None, opponent=None):
        self.kniffel = Kniffel()

    def render(self, mode='human', close=False):
        dice = self.kniffel.dice
        outfile = sys.stdout
        outfile.write(f'Dice: {dice[0]} {dice[1]} {dice[2]} {dice[3]} {dice[4]} '
                      f'Round: {self.kniffel.round}.{self.kniffel.sub_round} '
                      f'Score: {self.kniffel.get_total_score()}\n')
