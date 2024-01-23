from random import Random
from typing import Callable, Dict, List, Tuple, Union

from training.kniffelEngine.category import Category
from training.kniffelEngine.utils import (
    action_to_category_map,
    action_to_dice_roll_map,
    CATEGORY_ACTION_OFFSET,
    category_to_action_map,
    is_upper_section_category,
)
from training.kniffelEngine.scoring import (
    CONSTANT_SCORES,
    score_chance,
    score_full_house,
    score_large_straight,
    score_small_straight,
    score_upper_section,
    score_upper_section_bonus,
    score_x_of_a_kind,
    score_kniffel,
)


class Kniffel:
    def __init__(self, seed: int = None):
        # manage scores, rounds and dice
        self.scoring_functions: Dict[int, Callable[[List[int], bool], int]] = {}
        self.scores: Dict[Category, int] = {}
        self.init_scoring_functions()

        # a game has a total of 13 rounds
        self.round = 0

        # a round has 3 sub rounds: 2 dice rolls with choice and one scoring action
        self.sub_round = 0

        if seed:
            self.rnd = Random(seed)
        else:
            self.rnd = Random()

        # initialize dice
        self.dice = [0, 0, 0, 0, 0]
        self.roll_dice(True, True, True, True, True)

    def roll_dice(self, d1: bool, d2: bool, d3: bool, d4: bool, d5: bool):
        should_roll = [d1, d2, d3, d4, d5]
        for i, roll in enumerate(should_roll):
            if roll:
                self.dice[i] = self.rnd.choice([1, 2, 3, 4, 5, 6])

    def get_possible_actions(self) -> List[int]:
        possible_actions = []

        if self.round < 13:
            # determine if rerolling dice is possible; if so add all possible permutations
            if self.sub_round < 2:  # 2 times of dice rolling
                possible_actions.extend(list(range(CATEGORY_ACTION_OFFSET)))

            # choosing a scoring category is always possible
            for category in Category:
                if self.scores.get(category) is None:
                    action = category_to_action_map.get(category)
                    # Check if the category has an action associated with it
                    # (adding bonus is automatic).
                    if action:
                        possible_actions.append(action)

        return possible_actions

    def sample_action(self):
        actions = self.get_possible_actions()
        return self.rnd.sample(actions, 1)[0]

    def take_action(self, action: int) -> int:
        possible_actions = self.get_possible_actions()
        if action not in possible_actions:
            raise Exception('Action ' + str(action) + ' not allowed')

        # dice rolling action
        if action < CATEGORY_ACTION_OFFSET:
            self.sub_round += 1
            self.roll_dice(*action_to_dice_roll_map[action])
            return 0

        scores = self.get_action_score(action)
        for k, v in scores.items():
            old_score = self.scores.get(k, 0)
            if old_score:
                assert k == Category.KNIFFEL_BONUS  # only kniffel bonus can be changed more than once
            self.scores[k] = old_score + v

        # all non-rolling actions lead to the sub-round
        # ending and moving to the next round and rerolling all dice
        self.round += 1
        self.sub_round = 0
        self.roll_dice(True, True, True, True, True)
        return sum(scores.values())

    def is_eligible_for_kniffel_bonus(self):
        # rolls more than one kniffel
        return self.scores.get(Category.KNIFFEL, 0) > 0 and score_kniffel(self.dice)

    def is_finished(self):
        return self.round == 13

    def get_total_score(self):
        return sum([v for v in self.scores.values()])

    def get_action_score(self, action: int) -> Dict[Category, int]:
        category = action_to_category_map[action]
        scores: Dict[Category, int] = {}

        scoring_function = self.scoring_functions[category]
        if self.is_eligible_for_kniffel_bonus():    # Kniffel bonus and joker rule
            scores[Category.KNIFFEL_BONUS] = CONSTANT_SCORES[
                Category.KNIFFEL_BONUS
            ]
            scores[category] = scoring_function(self.dice, True)
        else:  # Regular rule
            if category == Category.FULL_HOUSE and score_kniffel(self.dice):  # kniffel could be used as full house
                scores[category] = scoring_function(self.dice, True)
            scores[category] = scoring_function(self.dice, False)

        # upper section bonus
        if is_upper_section_category(category):
            upper_scores = [
                v for k, v in self.scores.items() if is_upper_section_category(k)
            ]
            upper_scores.append(scores[category])
            if len(upper_scores) == 6:  # will only be executed once per game
                bonus_reward = score_upper_section_bonus(sum(upper_scores))
                scores[Category.UPPER_SECTION_BONUS] = bonus_reward

        return scores

    @staticmethod
    def get_action_description(action: int) -> str:
        if action < CATEGORY_ACTION_OFFSET:
            dice = action_to_dice_roll_map[action]
            rolled = [str(i + 1) for i, d in enumerate(dice) if d]
            return "Roll dice: " + " ".join(rolled)
        else:
            category = action_to_category_map[action]
            return f"Take {category.name}"

    def init_scoring_functions(self):
        self.scoring_functions[Category.ACES] = lambda x, y: score_upper_section(x, 1, joker=y)
        self.scoring_functions[Category.TWOS] = lambda x, y: score_upper_section(x, 2, joker=y)
        self.scoring_functions[Category.THREES] = lambda x, y: score_upper_section(x, 3, joker=y)
        self.scoring_functions[Category.FOURS] = lambda x, y: score_upper_section(x, 4, joker=y)
        self.scoring_functions[Category.FIVES] = lambda x, y: score_upper_section(x, 5, joker=y)
        self.scoring_functions[Category.SIXES] = lambda x, y: score_upper_section(x, 6, joker=y)
        self.scoring_functions[Category.THREE_OF_A_KIND] = lambda x, y: score_x_of_a_kind(x, 3, joker=y)
        self.scoring_functions[Category.FOUR_OF_A_KIND] = lambda x, y: score_x_of_a_kind(x, 4, joker=y)
        self.scoring_functions[Category.FULL_HOUSE] = lambda x, y: score_full_house(x, joker=y)
        self.scoring_functions[Category.SMALL_STRAIGHT] = lambda x, y: score_small_straight(x, joker=y)
        self.scoring_functions[Category.LARGE_STRAIGHT] = lambda x, y: score_large_straight(x, joker=y)
        self.scoring_functions[Category.KNIFFEL] = lambda x, y: score_kniffel(x)    # no joker rule, y as dummy
        self.scoring_functions[Category.CHANCE] = lambda x, y: score_chance(x, joker=y)
