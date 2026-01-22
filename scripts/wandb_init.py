import os

from dotenv import load_dotenv

import wandb

load_dotenv()

# Initialize new project
wandb.init(project="football-lstm", entity=os.getenv("WANDB_ENTITY"), name="initialize", mode="online")

wandb.finish()
