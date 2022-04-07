from data_loader import get_data
import models
from sklearn.metrics import r2_score
import torch
import wandb

wandb.init(project="CS637 Assignment 2", entity="jschwalb")
"""wandb.config = {
    "learning_rate": 0.00001,
    "epochs": 1000,
    "batch_size": 256
}"""
wandb.config = {
    "learning_rate": 0.001,
    "epochs": 500,
    "batch_size": 64
}

if __name__ == '__main__':
    timestamps, X_train, X_test, derived_x, y_train, y_test = get_data()

    train_X_chunks = [X_train[x:x + wandb.config['batch_size']] for x in range(0, len(X_train), wandb.config['batch_size'])]
    train_y_chunks = [y_train[x:x + wandb.config['batch_size']] for x in range(0, len(y_train), wandb.config['batch_size'])]

    test_X_chunks = [X_test[x:x + wandb.config['batch_size']] for x in range(0, len(X_test), wandb.config['batch_size'])]
    test_y_chunks = [y_test[x:x + wandb.config['batch_size']] for x in range(0, len(y_test), wandb.config['batch_size'])]

    #val_X_chunks = [val_X[x:x + wandb.config['batch_size']] for x in range(0, len(val_X), BATCH_SIZE)]
    #val_y_chunks = [val_y[x:x + wandb.config['batch_size']] for x in range(0, len(val_y), BATCH_SIZE)]

    in_dim = X_train.shape[1]
    model, loss_fn = models.first_model(in_dim=in_dim)
    #model, loss_fn = models.model_batch_norm_dropout(in_dim=in_dim)
    #model, loss_fn = models.model_batch_norm(in_dim=in_dim)
    #model, loss_fn = models.model_dropout(in_dim=in_dim)

    #model, loss_fn = models.second_model(in_dim=in_dim)
    #model, loss_fn = models.second_model_batch_norm_dropout(in_dim=in_dim)
    #model, loss_fn = models.second_model_batch_norm(in_dim=in_dim)
    #model, loss_fn = models.second_model_dropout(in_dim=in_dim)

    model.double()
    #wandb.watch(model, log="all", log_freq=100)
    wandb.watch(model, log="all")

    print(model)
    learning_rate = wandb.config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_losses = []
    test_losses = []
    model.train()
    for t in range(wandb.config['epochs']):
        print("Epoch: ", t)
        train_epoch_loss = 0
        test_epoch_loss = 0
        epoch_preds = []
        epoch_preds_test = []

        for i, (X_chunk, y_chunk, X_test_chunk, y_test_chunk) in enumerate(zip(train_X_chunks, train_y_chunks, test_X_chunks, test_y_chunks)):
            optimizer.zero_grad()
            y_pred = model(X_chunk)
            loss = loss_fn(y_pred, y_chunk)
            l0 = loss.item()
            train_epoch_loss += l0

            y_pred_test = model(X_test_chunk)
            loss_test = loss_fn(y_pred_test, y_test_chunk)
            l1 = loss_test.item()
            test_epoch_loss += l1

            #model.zero_grad()
            loss.backward()
            optimizer.step()
            """with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            """


            """
            if i % 100 == 99:
                y_pred_test = model(X_test_chunk)
                loss_test = loss_fn(y_pred_test, y_test_chunk)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        param -= learning_rate * param.grad

                l1 = loss_test.item()
                test_epoch_loss += l1
                #wandb.log({"batch_train_loss": l0 }, commit=False)
                #wandb.log({"batch_test_loss": l1}, commit=False)

                r0 = r2_score(y_chunk, y_pred.detach().numpy())
                r1 = r2_score(y_test_chunk, y_pred_test.detach().numpy())
                #wandb.log({"batch_train_r2_score": r0}, commit=False)
                #wandb.log({"batch_test_r2_score": r1})

            else:
                
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        param -= learning_rate * param.grad
            """
        wandb.log({"epoch_train_loss": train_epoch_loss}, commit=False)
        wandb.log({"epoch_test_loss": test_epoch_loss})
        #wandb.log({"epoch_train_r2": r2_score(y_true=y_train, y_pred=epoch_preds)}, commit=False)
        #wandb.log({"epoch_test_r2": r2_score(y_true=y_test, y_pred=epoch_preds_test)})

