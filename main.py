from data_loader import get_data
from models import first_model
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
    model, loss_fn = first_model(in_dim=in_dim)
    model.double()
    print(model)
    learning_rate = wandb.config['learning_rate']

    train_losses = []
    test_losses = []

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


            l0 = loss.item()
            l1 = loss_val.item()
            wandb.log({"train_loss": l0})
            wandb.log({"test_loss": l1})

            r0 = r2_score(y_train, y_pred.detach().numpy())
            r1 = r2_score(y_test, y_pred_test.detach().numpy())
            wandb.log({"Train_r2_score": r0})
            wandb.log({"Test_r2_score": r1})

            #wandb.log({"averaged_train_loss": l0/X_train.shape[0]})
            #wandb.log({"averaged_test_loss": l1/X_test.shape[0]})

            # Optional
            wandb.watch(model)

            #train_losses.append(l0)
            #test_losses.append(l1)

            #print(t, l0)

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # You can access the first layer of `model` like accessing the first item of a list
    linear_layer = model[0]

    print("Hello World")
