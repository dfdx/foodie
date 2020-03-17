import dill
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.distance import CosineSimilarity
from torchvision import models
from torchvision import transforms
from foodie.datasets import TripletImageFolder
from foodie.utils import train_test_loader
from foodie.losses import TripletLoss


FOOD101_IMG_PATH = Path("~/data/food-101/images").expanduser()
OWN_IMG_PATH = Path("~/data/foodie/own").expanduser()
MODEL_PATH = Path("~/models/foodie_e86.pkl").expanduser()
OWN_MODEL_PATH = Path("~/models/foodie_own.pkl").expanduser()


device = (torch.device('cuda')
          if torch.cuda.is_available()
          else torch.device('cpu'))


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.eval()
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    def forward(self, x):
        y = self.model.forward(x)
        y = y.view(y.size(0), -1)
        y = F.normalize(y, p=2, dim=1)
        return y

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


def train(model, train_set, loss_function, epoch, optimizer=None):
    print(f"==== Epoch: {epoch} ====")
    model.train()
    losses = []
    for i, xs in enumerate(train_set, 1):
        xs = [x.to(device) for x in xs]
        model.zero_grad()
        emb = model(*xs)
        loss = loss_function(*emb)
        losses.append(loss.item())
        if i % 100 == 0:
            print(f'Epoch {epoch}; iter {i}/{len(train_set.dataset)}; '
                  f'avg epoch loss = {np.mean(losses)}')
        loss.backward()
        optimizer.step()
    print(f"Avg loss: {np.mean(losses)}")


def test(model, test_set, loss_function, epoch, optimizer=None):
    print(f"==== Testing ====")
    model.eval()
    losses = []
    for i, xs in enumerate(test_set, 1):
        xs = [x.to(device) for x in xs]
        with torch.no_grad():
            model.zero_grad()
            emb = model(*xs)
            loss = loss_function(*emb)
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    print(f"Avg loss: {avg_loss}")
    return avg_loss


def train_test_triplet():
    dataset = TripletImageFolder(OWN_IMG_PATH)
    train_set, test_set = train_test_loader(dataset, batch_size=1)
    model = TripletNet(EmbeddingNet()).to(device)
    loss_function = TripletLoss(1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    min_test_loss = np.inf
    for epoch in range(10):
        train(model, train_set, loss_function, epoch, optimizer=optimizer)
        if epoch % 2 == 0:
            test_loss = test(model, test_set, loss_function,
                             epoch, optimizer=optimizer)
            if test_loss < min_test_loss:
                print(f"New test loss is {test_loss}, which is smaller "
                      f"than {min_test_loss}. Saving the model.")
                model.embedding_net.save(OWN_MODEL_PATH)
                min_test_loss = test_loss
