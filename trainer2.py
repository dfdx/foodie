import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import setup_custom_logger


torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = setup_custom_logger("foodie")


def train(model, trainset, testset, loss_function, optimizer=None, n_epochs=100, tboard=False, saveto=None):
    optimizer = optimizer or optim.SGD(model.parameters(), lr=1e-6)
    if tboard:
        # don't forget to run `tensorboard --logdir=logs` from mdn dir
        tensorboard = Tensorboard("mdn/logs")
    # min_error = test(model, testset, loss_function)
    for epoch in range(n_epochs):
        print(f"==== Epoch: {epoch} ====")
        batch_losses = []
        for i, xs in enumerate(trainset, 1):
            xs = [x.to(device) for x in xs]
            # reset gradients and hidden state
            model.zero_grad()           
            # run forward pass
            emb = model(*xs)
            # compute the loss and backpropagate gradients
            loss = loss_function(*emb)
            if tboard:
                tensorboard.log_scalar("loss", loss.item(), epoch * len(trainset.dataset) + i)
            batch_losses.append(loss.item())
            if i % 100 == 0:
                log.info(f'Epoch {epoch}; iter {i}/{len(trainset.dataset)}; '
                         f'avg batch loss = {np.mean(batch_losses)}')
                batch_losses = []
            loss.backward()
            optimizer.step()
        print(f"Avg loss: {np.mean(batch_losses)}")    
        # if (epoch + 1) % 5 == 0:
        #     error = test(model, trainset, loss_function)
        #     log.info(f"On train data: loss = {error}")
        #     error = test(model, testset, loss_function)
        #     log.info(f"On test data: loss = {error}")
        #     if error < min_error:
        #         log.info("New min error, saving the model")
        #         min_error = error
        #         if saveto:
        #             torch.save(model, saveto, pickle_module=dill)
    return model


# # this might be faster than test(), but it has a number of restrictions
# def batch_test(model, testset, loss_function):
#     ys = []
#     y_hats = []
#     with torch.no_grad():
#         for i, obs in enumerate(testset):
#             x_static, x_events, y = [x.squeeze(0).to(device) for x in obs]
#             model.hidden = model.init_hidden()
#             y_hat = model(x_static, x_events)
#             y_hats.append(y_hat.item())
#             ys.append(y.item())
#     loss = loss_function(torch.FloatTensor(y_hats), torch.FloatTensor(ys))
#     return loss.item()


# def test(model, testset, loss_function):
#     losses = []
#     with torch.no_grad():
#         for i, obs in enumerate(testset):
#             x_static, x_events, y = [x.squeeze(0).to(device) for x in obs]
#             model.hidden = model.init_hidden()
#             y_hat = model(x_static, x_events)
#             loss = loss_function(y_hat, y).item()
#             losses.append(loss)
#     return np.mean(losses)
