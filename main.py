from data_loader import get_data
import models
from sklearn.metrics import r2_score
import torch
import wandb

wandb.init(project="CS637 Assignment 2", entity="jschwalb")
wandb.config = {
    "learning_rate": 0.00001,
    "epochs": 10000,
    "batch_size": 128
}

if __name__ == '__main__':
    timestamps, X_train, X_test, derived_x, y_train, y_test = get_data()

    in_dim = X_train.shape[1]
    #model, loss_fn = models.first_model(in_dim=in_dim)
    #model, loss_fn = models.model_batch_norm_dropout(in_dim=in_dim)
    #model, loss_fn = models.model_batch_norm(in_dim=in_dim)
    model, loss_fn = models.model_dropout(in_dim=in_dim)

    #models = [model0, model1, model2, model3]
    #loss_fns = [loss_fn0, loss_fn1, loss_fn2, loss_fn3]
    #for model, loss_fn in zip(models, loss_fns):

    model.double()
    wandb.watch(model, log="all", log_freq=100)

    print(model)
    learning_rate = wandb.config['learning_rate']

    train_losses = []
    test_losses = []
    model.train()
    for t in range(wandb.config['epochs']):
        #for item_i, (train_item, train_label) in enumerate(zip(X_train, y_train)):
        # Forward pass: compute predicted y by passing x to the model. Module objects

        y_pred = model(X_train)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y_train)


        if t % 100 == 99:
            y_pred_test = model(X_test)
            loss_val = loss_fn(y_pred_test, y_test)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

            l0 = loss.item()
            l1 = loss_val.item()
            wandb.log({"train_loss": l0}, commit=False)
            wandb.log({"test_loss": l1}, commit=False)

            r0 = r2_score(y_train, y_pred.detach().numpy())
            r1 = r2_score(y_test, y_pred_test.detach().numpy())
            wandb.log({"Train_r2_score": r0}, commit=False)
            wandb.log({"Test_r2_score": r1})

        else:
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

