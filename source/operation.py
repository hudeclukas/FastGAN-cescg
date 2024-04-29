import os
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import shutil
import json


def InfiniteSampler(n):
    """Data sampler"""
    # check if the number of samples is valid
    if n <= 0:
        raise ValueError(
            f"Invalid number of samples: {n}.\nMake sure that images are present in the given path."
        )
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2**31


def copy_Generator_parameters(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_parameters(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def get_dir(args):

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    task_name = os.path.join(args.output_path, "train_results", args.name)
    saved_model_folder = os.path.join(task_name, "models")
    saved_image_folder = os.path.join(task_name, "images")

    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    for f in os.listdir("./"):
        if ".py" in f:
            shutil.copy(f, os.path.join(task_name, f))

    with open(os.path.join(saved_model_folder, "../args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder


class ImageFolder(Dataset):
    """ArtDataset"""

    def __init__(self, root, transform=None):
        super(ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if (
                image_path[-4:] == ".jpg"
                or image_path[-4:] == ".png"
                or image_path[-5:] == ".jpeg"
            ):
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
