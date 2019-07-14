from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.distance import CosineSimilarity
from torchvision import models
from datasets import TripletImageFolder
from utils import imshow, train_test_loader
from trainer2 import train
from losses import TripletLoss


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class EmbeddingResnet(nn.Module):
    def __init__(self):
        super(EmbeddingResnet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.eval()
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    def forward(self, x):
        y = self.model.forward(x)
        y = y.view(y.size(0), -1)
        y = F.normalize(y, p=2, dim=1)
        return y


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



def main_old():
    embed = EmbeddingResnet().to(device)
    x = torch.rand(1, 3, 214, 214).to(device)
    y = embed(x)


FOOD101_IMG_PATH = Path("~/data/food-101/images").expanduser()


def try_untrained():
    dataset = TripletImageFolder(FOOD101_IMG_PATH)
    embed = EmbeddingResnet().to(device)
    sim = CosineSimilarity()
    a, b, c = map(lambda x: x.unsqueeze(0).to(device), dataset[1001])
    print(f"sim(a,b) = {sim(embed(a), embed(b)).item()}")
    print(f"sim(a,c) = {sim(embed(a), embed(c)).item()}")
    print(f"sim(b,c) = {sim(embed(b), embed(c)).item()}")


def main():
    dataset = TripletImageFolder(FOOD101_IMG_PATH)
    trainset, testset = train_test_loader(dataset, batch_size=1)
    model = TripletNet(EmbeddingResnet()).to(device)
    loss_function = TripletLoss(1.0)
    train(model, trainset, testset, loss_function)
