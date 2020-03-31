import heapq
from pathlib import Path
from PIL import Image
import torch
from torch.nn import CosineSimilarity
from torchvision import transforms
from foodie.models import EmbeddingNet, OWN_MODEL_PATH


device = (torch.device('cuda')
          if torch.cuda.is_available()
          else torch.device('cpu'))


IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
])


def load_preprocess_image(path):
    img = Image.open(path)
    t = IMG_TRANSFORM(img)
    return t


class ImageIndex:

    def __init__(self, embed_model_path=OWN_MODEL_PATH):
        self.embed = EmbeddingNet()
        self.embed.load(embed_model_path).to(device)
        self.embeddings = []
        self.labels = []
        self.paths = []

    def load(self, index_path):
        self.embeddings = []
        self.labels = []
        self.paths = []
        with open(index_path) as f:
            for line in f:
                e_str, label, path = line.split("\t")
                e = torch.FloatTensor(e_str.split(","))
                self.embeddings.append(e)
                self.labels.append(label)
                self.paths.append(path)
        return self

    def load_from_dir(self, img_dir):
        for subdir in Path(img_dir).glob("*"):
            label = subdir.stem
            for path in subdir.glob("*"):
                t = load_preprocess_image(path)
                self.add(t, label, path)

    def save(self, index_path):
        with open(index_path, "w") as f:
            for e, label, path in zip(self.embeddings, self.labels, self.paths):  # noqa: E501
                e_str = ",".join(x for x in e)
                self.embeddings.append(e)
                self.labels.append(label)
                self.paths.append(path)
                f.write(f"{e_str}\t{label}\t{path}")

    def get_embedding(self, t):
        if len(t.shape) == 3:
            t = t.unsqueeze(0)
        with torch.no_grad():
            e = self.embed(t.to(device)).squeeze(0)
        return e

    def add(self, t, label, path):
        e = self.get_embedding(t)
        self.embeddings.append(e.unsqueeze(0))
        self.labels.append(label)
        self.paths.append(str(path))

    def find_similar(self, t, n=5):
        e = self.get_embedding(t)
        sim = CosineSimilarity()
        similarities = [sim(e, se).item() for se in self.embeddings]
        top = heapq.nlargest(5, enumerate(similarities), key=lambda p: p[1])
        return top
