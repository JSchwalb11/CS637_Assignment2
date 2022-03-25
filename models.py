import torch.nn as nn
from data_loader import get_data

def first_model(in_dim):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.ReLU(),
        nn.Linear(25, 34),
        nn.Sigmoid(),
        nn.Linear(34, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn


if __name__ == '__main__':
    model, loss_rn = first_model()