
import torch

model = torch.load("models/bert")
trainer = torch.load("models/trainer")

torch.save(model.state_dict(), "models/bert")
torch.save(trainer.state_dict(), "models/trainer")

