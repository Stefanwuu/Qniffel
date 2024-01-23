from collections import defaultdict
from enum import Enum
import logging
import sys
from typing import Dict, Optional, Sequence

from gym import Env, spaces

import numpy as np

from training.kniffelEngine import Kniffel, Category


log = logging.getLogger(__name__)


def get_score(score: Optional[int]) -> np.ndarray:
    return np.array([score], dtype=np.int16) if score is not None else np.array([-1], dtype=np.int16)


class KniffelSingleEnv(Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, seed=None, render_mode='human'):
        self.kniffel = Kniffel(seed=seed)
        self.action_space = spaces.Discrete(44)
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.observation_space = spaces.Tuple((
            spaces.Discrete(13),  # round
            spaces.Discrete(3),  # sub-round
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
            spaces.Box(low=-1, high=600, shape=(1,), dtype=np.int16),  # kniffel bonus
        ))

    def get_observation_space(self):
        kniffel = self.kniffel
        return (
            kniffel.round,
            kniffel.sub_round,
            np.array([kniffel.dice[0]], dtype=np.uint8),
            np.array([kniffel.dice[1]], dtype=np.uint8),
            np.array([kniffel.dice[2]], dtype=np.uint8),
            np.array([kniffel.dice[3]], dtype=np.uint8),
            np.array([kniffel.dice[4]], dtype=np.uint8),
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
        # agent should avoid out of rule action by itself: otherwise exception raised
        reward = kniffel.take_action(action)
        finished = kniffel.is_finished()
        valid_move = True

        log.info(f'Finished step. Reward: {reward}, Finished: {finished}')
        debug_info = {
            'valid_move': valid_move,
            'action_description': kniffel.get_action_description(action)
        }
        return self.get_observation_space(), reward, finished, False, debug_info

    def reset(self, seed=None, opponent=None, options=None):
        self.kniffel = Kniffel()
        return self.get_observation_space(), {}

    def render(self):
        assert self.render_mode == 'human'
        dice = self.kniffel.dice
        outfile = sys.stdout
        outfile.write(f'Dice: {dice[0]} {dice[1]} {dice[2]} {dice[3]} {dice[4]} '
                      f'Round: {self.kniffel.round}.{self.kniffel.sub_round} '
                      f'Score: {self.kniffel.get_total_score()}\n')
