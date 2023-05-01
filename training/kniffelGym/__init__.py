from gym.envs.registration import register

register(
    id='kniffel-single-v0',
    entry_point='training.kniffelGym.envs:KniffelSingleEnv',
)
