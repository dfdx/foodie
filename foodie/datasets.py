import glob
import random
from PIL import Image
from pathlib import Path
import torch
from torch.utils import data
from torchvision import transforms
from foodie.utils import imshow


def default_image_loader(path):
    return Image.open(path).convert('RGB')


def choose_random_not_equal_to(lst, x):
    y = x
    for cnt in range(1000):  # 1000 attempts to choose a good y not equal to x
        y = random.choice(lst)
        if x != y:
            return y
    raise Exception(f"Could not choose a random object not equal to {x} after 1000 attempts")


class TripletImageFolder(torch.utils.data.Dataset):
    def __init__(self, base_path, transform=None, loader=default_image_loader):
        self.base_path = Path(base_path)
        self.imgs = {}
        for path in self.base_path.glob("*"):
            label = path.stem
            such_imgs = []
            for img_path in path.glob("*"):
                such_imgs.append(img_path)
            self.imgs[label] = such_imgs
        self.triplets = []
        labels = list(self.imgs.keys())
        for label in labels:
            img_paths = list(self.imgs[label])
            for img_path in img_paths:
                a = img_path
                # choose random image with the same label
                b = choose_random_not_equal_to(img_paths, img_path)
                # choose random image with another label
                c_label = choose_random_not_equal_to(labels, label)
                c = random.choice(self.imgs[c_label])
                self.triplets.append((a, b, c))
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
        ])
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = map(str, self.triplets[index])
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        img3 = self.loader(path3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)


def main():
    base_path = Path("~/data/food-101/images").expanduser()
    ds = TripletImageFolder(base_path)
