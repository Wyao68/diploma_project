import inspect
import torch
print(inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__))