import pprint
from __init__ import Kniffel

start_new_game = True


def user_choice(choices: list):
    choice = input("Choose an action from above: ")
    while choice not in choices:
        choice = input("Wrong input, choose an action: ")
    return choice


while start_new_game:
    game = Kniffel()
    while not game.is_finished():
        print("Round: ", game.round + 1)
        print("Sub-Round: ", game.sub_round + 1)
        print("Scores: ", game.scores)
        print("Dice: ", game.dice)
        input_action_map = {}
        for action in game.get_possible_actions():
            input_action_map[Kniffel.get_action_description(action)] = action
        print("Possible actions: ")
        pprint.pprint(input_action_map)
        action = user_choice(list(map(str, game.get_possible_actions())))
        game.take_action(int(action))

    pprint.pprint(game.scores)
    print("Final scores: ", sum(game.scores))
    print("Game finished! Do you want another game? (y/n)")
    start_new_game = user_choice(['y', 'n']) == 'y'
