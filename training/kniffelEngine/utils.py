"""
Common utilities needed that make it possible to convert actions (e.g. choosing
threes on the scorecard) to categories (e.g. full house on the scorecard) and vice versa.
Similar maps are also provided for mapping dice rolling lists (e.g. reroll dice 2, 3 and
4 but keep dice 1 and 5) to and from actions and categories to scoring functions.
"""
from typing import Dict, List, Tuple

from training.kniffelEngine.category import Category

# Mapping from action id to permutations of rerolling of dice. Each unique combination
# of dice rolls is given a unique id, resulting in 31 unique constellations(all 0 not allowed).
# An offset constant is provided, as category actions are located after dice rolling actions.
CATEGORY_ACTION_OFFSET = 31
action_to_dice_roll_map: Dict[int, Tuple[bool, bool, bool, bool, bool]] = {}
dice_roll_to_action_map: Dict[Tuple[bool, bool, bool, bool, bool], int] = {}
for d1 in [1, 0]:
    for d2 in [1, 0]:
        for d3 in [1, 0]:
            for d4 in [1, 0]:
                for d5 in [1, 0]:
                    # make rolling all dice the first action, i.e. zero
                    key = CATEGORY_ACTION_OFFSET - (
                        d5 * 2 ** 0
                        + d4 * 2 ** 1
                        + d3 * 2 ** 2
                        + d2 * 2 ** 3
                        + d1 * 2 ** 4
                    )
                    value = bool(d1), bool(d2), bool(d3), bool(d4), bool(d5)
                    # not rolling any dice(d1-d5 all zero) is not a valid action
                    if key < 31:
                        action_to_dice_roll_map[key] = value
                        dice_roll_to_action_map[value] = key


# Mapping from action id to category and vice versa.
action_to_category_map: Dict[int, Category] = {}
category_to_action_map: Dict[Category, int] = {}
for i, category in enumerate(Category):
    if category in (Category.UPPER_SECTION_BONUS, Category.KNIFFEL_BONUS):
        continue    # skip the two automatic categories
    action_to_category_map[i + CATEGORY_ACTION_OFFSET] = category
    category_to_action_map[category] = i + CATEGORY_ACTION_OFFSET

# List of actionable categories e.g. for determining valid actions
actionable_categories: List[Category] = []
for category in Category:
    if category not in (Category.UPPER_SECTION_BONUS, Category.KNIFFEL_BONUS):
        actionable_categories.append(category)


def is_upper_section_category(category: Category) -> bool:
    return int(category) < 6

