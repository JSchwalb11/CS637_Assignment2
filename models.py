import torch.nn as nn
from data_loader import get_data

def first_model(in_dim, batch_norm=True, dropout=True, p=0.2):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.LayerNorm(25),
        nn.ReLU(),
        nn.Linear(25, 34),
        nn.LayerNorm(34),
        nn.Sigmoid(),
        nn.Linear(34, 4),
        nn.LayerNorm(4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn

def model_batch_norm_dropout(in_dim, p=0.2):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.BatchNorm1d(25),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(25, 34),
        nn.BatchNorm1d(34),
        nn.Sigmoid(),
        nn.Dropout(p),
        nn.Linear(34, 4),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn

def model_dropout(in_dim, p=0.2):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.LayerNorm(25),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(25, 34),
        nn.LayerNorm(34),
        nn.Sigmoid(),
        nn.Dropout(p),
        nn.Linear(34, 4),
        nn.LayerNorm(4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn

def model_batch_norm(in_dim, p=0.2):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.BatchNorm1d(25),
        nn.ReLU(),
        nn.Linear(25, 34),
        nn.BatchNorm1d(34),
        nn.Sigmoid(),
        nn.Linear(34, 4),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn

def second_model(in_dim, batch_norm=True, dropout=True, p=0.2):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.LayerNorm(25),
        nn.ReLU(),
        nn.Linear(25, 34),
        nn.LayerNorm(34),
        nn.Sigmoid(),
        nn.Linear(34, 19),
        nn.LayerNorm(19),
        nn.ReLU(),
        nn.Linear(19, 12),
        nn.LayerNorm(12),
        nn.Sigmoid(),
        nn.Linear(12, 4),
        nn.LayerNorm(4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn

def second_model_batch_norm_dropout(in_dim, p=0.2):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.BatchNorm1d(25),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(25, 34),
        nn.BatchNorm1d(34),
        nn.Sigmoid(),
        nn.Dropout(p),
        nn.Linear(34, 19),
        nn.BatchNorm1d(19),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(19, 12),
        nn.BatchNorm1d(12),
        nn.Sigmoid(),
        nn.Dropout(p),
        nn.Linear(12, 4),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn

def second_model_dropout(in_dim, p=0.2):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.LayerNorm(25),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(25, 34),
        nn.LayerNorm(34),
        nn.Sigmoid(),
        nn.Dropout(p),
        nn.Linear(34, 19),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(19, 12),
        nn.Sigmoid(),
        nn.Dropout(p),
        nn.Linear(12, 4),
        nn.LayerNorm(4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn

def second_model_batch_norm(in_dim, p=0.2):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.BatchNorm1d(25),
        nn.ReLU(),
        nn.Linear(25, 34),
        nn.BatchNorm1d(34),
        nn.Sigmoid(),
        nn.Linear(34, 19),
        nn.BatchNorm1d(19),
        nn.ReLU(),
        nn.Linear(19, 12),
        nn.BatchNorm1d(12),
        nn.Sigmoid(),
        nn.Linear(12, 4),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn

def MLP(in_dim):
    model = nn.Sequential(
        nn.Linear(in_dim, 25),
        nn.ReLU(),
        nn.Linear(25, 34),
        nn.Sigmoid(),
        nn.Linear(34, 14),
        nn.ReLU(),
        nn.Linear(14, 16),
        nn.Sigmoid(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 54),
        nn.Sigmoid(),
        nn.Linear(54, 66),
        nn.ReLU(),
        nn.Linear(66, 31),
        nn.Sigmoid(),
        nn.Linear(31, 13),
        nn.ReLU(),
        nn.Linear(13, 5),
        nn.Sigmoid(),
        nn.Linear(5, 1)
    )

    model = model.float()
    loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = nn.MSELoss(reduction='sum')

    return model, loss_fn


if __name__ == '__main__':
    model, loss_rn = first_model()