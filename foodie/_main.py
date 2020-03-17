import dill
from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.distance import CosineSimilarity
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision import transforms
from foodie.datasets import TripletImageFolder
from foodie.utils import imshow, train_test_loader
from foodie.trainer2 import train
from foodie.losses import TripletLoss


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


# TODO:
# 1. move models to foodie.model
# 2. use TripletNet for trainin, but only EmbeddingNet for embedding
# 3. add utilities for transorming images during loading + check/transforming during embedding

    
def main_old():
    embed = EmbeddingResnet().to(device)
    x = torch.rand(1, 3, 214, 214).to(device)
    y = embed(x)





def try_untrained():
    # dataset = TripletImageFolder(FOOD101_IMG_PATH)
    dataset = TripletImageFolder(OWN_IMG_PATH)
    embed = EmbeddingResnet().to(device)
    sim = CosineSimilarity()
    a, b, c = map(lambda x: x.unsqueeze(0).to(device), dataset[2])
    print(f"sim(a,b) = {sim(embed(a), embed(b)).item()}")
    print(f"sim(a,c) = {sim(embed(a), embed(c)).item()}")
    print(f"sim(b,c) = {sim(embed(b), embed(c)).item()}")


def main():
    dataset = TripletImageFolder(OWN_IMG_PATH)
    trainset, testset = train_test_loader(dataset, batch_size=1)
    model = TripletNet(EmbeddingResnet()).to(device)
    loss_function = TripletLoss(1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train(model, trainset, testset, loss_function, optimizer=optimizer)
    # MODEL_PATH.parents[0].mkdirs(exist_ok=True)
    # torch.save(model, MODEL_PATH, pickle_module=dill)


def similarity(model, dataset, i, j):
    sim = CosineSimilarity()
    a_img = dataset[i][0]
    b_img = dataset[j][0]
    imshow(a_img)
    imshow(b_img)
    a = model.get_embedding(a_img.unsqueeze(0).to(device))
    b = model.get_embedding(b_img.unsqueeze(0).to(device))
    print(f"sim(a,b) = {sim(a, b).item()}")
    
    
def try_trained():
    # dataset = TripletImageFolder(FOOD101_IMG_PATH)
    dataset = ImageFolder(FOOD101_IMG_PATH, transform=transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
        ]))
    model = torch.load(MODEL_PATH, pickle_module=dill)
    sim = CosineSimilarity()
    a, b, c = map(lambda x: x.unsqueeze(0).to(device), [dataset[95000][0], dataset[93003][0], dataset[93004][0]])
    a, b, c = map(model.get_embedding, (a, b, c))
    print(f"sim(a,b) = {sim(a, b).item()}")
    print(f"sim(a,c) = {sim(a, c).item()}")
    print(f"sim(b,c) = {sim(b, c).item()}")


# def train(model, trainset, testset, loss_function, optimizer=None, n_epochs=100, tboard=False, saveto=None):
#     optimizer = optimizer or optim.SGD(model.parameters(), lr=1e-6)
#     if tboard:
#         # don't forget to run `tensorboard --logdir=logs` from this dir
#         tensorboard = Tensorboard("tboard/logs")
#     # min_error = test(model, testset, loss_function)
#     for epoch in range(n_epochs):
#         batch_losses = []
#         for i, xs in enumerate(trainset, 1):
#             xs = [x.to(device) for x in xs]
#             # reset gradients and hidden state
#             model.zero_grad()           
#             # run forward pass
#             emb = model(*xs)
#             # compute the loss and backpropagate gradients
#             loss = loss_function(*emb)
#             if tboard:
#                 tensorboard.log_scalar("loss", loss.item(), epoch * len(trainset.dataset) + i)
#             batch_losses.append(loss.item())
#             if i % 100 == 0:
#                 log.info(f'Epoch {epoch}; iter {i}/{len(trainset.dataset)}; '
#                          f'avg batch loss = {np.mean(batch_losses)}')
#                 batch_losses = []
#             loss.backward()
#             optimizer.step()
#         # if (epoch + 1) % 5 == 0:
#         #     error = test(model, trainset, loss_function)
#         #     log.info(f"On train data: loss = {error}")
#         #     error = test(model, testset, loss_function)
#         #     log.info(f"On test data: loss = {error}")
#         #     if error < min_error:
#         #         log.info("New min error, saving the model")
#         #         min_error = error
#         #         if saveto:
#         #             torch.save(model, saveto, pickle_module=dill)
#     return model


# def main2():
    
