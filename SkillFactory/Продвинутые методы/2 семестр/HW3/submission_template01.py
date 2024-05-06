import numpy as np
import torch
from torch import nn

class create_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 16, bias=False)
        self.fc3 = nn.Linear(16, 10, bias=False)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
model = create_model()


def count_parameters(model):
    # your code here
    # return integer number (None is just a placeholder)
    from prettytable import PrettyTable
    # your code here
    # return integer number (None is just a placeholder)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    