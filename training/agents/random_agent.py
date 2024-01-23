from training.agents.agent import agent
import wandb

class random(agent):
    def __init__(self, env, config):
        super().__init__(env, config)
        wandb.init(
            project="Qniffel",
            group="random",
        )

    def decide(self, state):
        return self.env.sample_action()