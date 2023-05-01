from collections import Counter
from typing import Dict, List, Set

from training.kniffelEngine.category import Category

CONSTANT_SCORES: Dict[Category, int] = {
    Category.FULL_HOUSE: 25,
    Category.SMALL_STRAIGHT: 30,
    Category.LARGE_STRAIGHT: 40,
    Category.KNIFFEL: 50,
    Category.KNIFFEL_BONUS: 50,
    Category.UPPER_SECTION_BONUS: 35,
}


def score_upper_section(dice: List[int], face: int, joker: bool = False) -> int:
    # joker returns the maximum possible score
    return len(dice) * face if joker else sum(die if die == face else 0 for die in dice)


def score_x_of_a_kind(dice: List[int], min_same_faces: int, joker: bool = False) -> int:
    """Return sum of dice if there are a minimum of equal min_same_faces dice, otherwise
    return zero. Only works for 3 or more min_same_faces.
    """
    if joker:
        return len(dice) * 6  # maximum possible score
    for die, count in Counter(dice).most_common(1):
        if count >= min_same_faces:
            return sum(dice)
    return 0


def score_full_house(dice: List[int], joker: bool = False) -> int:
    global CONSTANT_SCORES
    counter = Counter(dice)
    if (len(counter.keys()) == 2 and min(counter.values()) == 2) or joker:
        return CONSTANT_SCORES[Category.FULL_HOUSE]
    return 0


def _are_two_sets_equal(a: Set, b: Set) -> bool:
    return a.intersection(b) == a


def score_small_straight(dice: List[int], joker: bool = False) -> int:
    """
    Small straight scoring
    """
    global CONSTANT_SCORES
    dice_set = set(dice)
    if (
        _are_two_sets_equal({1, 2, 3, 4}, dice_set)
        or _are_two_sets_equal({2, 3, 4, 5}, dice_set)
        or _are_two_sets_equal({3, 4, 5, 6}, dice_set)
        # large straight can be used as small straight
        or _are_two_sets_equal({1, 2, 3, 4, 5}, dice_set)
        or _are_two_sets_equal({2, 3, 4, 5, 6}, dice_set)
    ) or joker:
        return CONSTANT_SCORES[Category.SMALL_STRAIGHT]
    return 0


def score_large_straight(dice: List[int], joker: bool = False) -> int:
    """
    Large straight scoring
    """
    global CONSTANT_SCORES
    dice_set = set(dice)
    if (
        _are_two_sets_equal({1, 2, 3, 4, 5}, dice_set)
        or _are_two_sets_equal({2, 3, 4, 5, 6}, dice_set)
    ) or joker:
        return CONSTANT_SCORES[Category.LARGE_STRAIGHT]
    return 0


def score_kniffel(dice: List[int]) -> int:
    global CONSTANT_SCORES
    if len(set(dice)) == 1:
        return CONSTANT_SCORES[Category.KNIFFEL]
    return 0


def score_chance(dice: List[int], joker: bool = False) -> int:
    if joker:
        return len(dice) * 6  # maximum possible score
    return sum(dice)


def score_upper_section_bonus(upper_section_score: int) -> int:
    global CONSTANT_SCORES
    if upper_section_score >= 63:
        return CONSTANT_SCORES[Category.UPPER_SECTION_BONUS]
    return 0
