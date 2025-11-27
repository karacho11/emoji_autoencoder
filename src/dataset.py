import numpy as np
from typing import List, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from datasets import load_dataset

# Keyword filtering
KEYWORDS = [
    "face", "christmas", "superhero", "supervillian", "mage", "vampire",
    "monkey", "elf", "juggling", "boy", "girl", "adult", "person", "man",
    "woman", "male", "female", "worker", "scientist", "technologist",
    "singer", "artist", "pilot", "astronaut", "firefighter", "police",
    "sleuth", "construction"
]

def load_and_filter_data(
        image_size: int=64
):
    """
    Load vahalla/emoji-dataset
    filter by KEYWORDS
    resize to image_size -> recommended to 64*64*3

    Returns: [x_unshuffled (N, C, H, W), text_unshuffled, x_tensor (N, C, H, W), texts]
    """

    ds = load_dataset("vahalla/emoji-dataset")
    train_split = ds["train"]

    selected_images = []
    selected_texts = []

    for i in range(len(train_split)):
        im = train_split[i]["image"]
        text = train_split[i]["text"]

        if any(kw in text for kw in KEYWORDS):
            im = im.resize((image_size, image_size), Image.LANCZOS)
            arr = np.array(im, dtype=np.float32)            # H * W * C
            arr = np.transpose(arr, (2, 0, 1))              # C * H * W
            selected_images.append(arr)
            selected_texts.append(text)

    data = np.stack(selected_images)                        # N * C * H * W
    x_unshuffled = torch.from_numpy(data)                   # keep a copy of unshuffled, will use later

    x_tensor = x_unshuffled.clone()                         # tensor that we will shuffle later
    texts = list(selected_texts)

    return x_unshuffled, selected_texts, x_tensor, texts



def split_data(
        x: torch.Tensor,
        texts: List[str],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
):
    """
    Shuflfle and split data -- recommended: 60/20/20

    Returns: x_train, x_val, x_test, texts_train, texts_val, texts_test
    """

    assert abs(train_ratio + val_ratio - 0.8) < 1e-6, "Train+Val should be 0.8!"

    N = x.shape[0]
    indices = torch.randperm(N)
    x = x[indices]
    texts = [texts[i] for i in indices.tolist()]

    n_train = int(train_ratio * N)
    n_val = int(val_ratio * N)
    n_test = N - n_train - n_val

    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]

    texts_train = texts[:n_train]
    texts_val = texts[n_train:n_train + n_val]
    texts_test = texts[n_train + n_val:]

    return x_train, x_val, x_test, texts_train, texts_val, texts_test


class EmojiDataset(Dataset):
    """
    Dataset wrapper 
    
    data_tensor: (N, C, H, W) float32 in [0,1]
    texts
    transform: torchvision transform -- (e.g., RandomHorizontalFlip)
    """

    def __init__(self, data_tensor: torch.Tensor, texts: List[str], transform=None):
        
        assert data_tensor.shape[0] == len(texts), "data_tensor and texts must have same length!"
        self.data = data_tensor
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx:int):
        img = self.data[idx]            # C * H * W
        text = self.texts[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, text
    


def get_dataloaders(
        batch_size: int=32,
        image_size: int=64,
        augment: bool=True
):
    """
    Load and filters emoji data, splits into train/val/test

    Returns: train_dl, val_dl, test_dl, x_unshuffled, texts_unshuffled
    """

    # Load data
    x_unshuffled, texts_unshuffled, x, texts = load_and_filter_data(image_size=image_size)

    # Split
    x_train, x_val, x_test, texts_train, texts_val, texts_test = split_data(x, texts, train_ratio=0.6, val_ratio=0.2)

    # Data augmentation
    train_transform = None
    if augment:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
        ])

    train_ds = EmojiDataset(x_train, texts_train, train_transform)
    val_ds = EmojiDataset(x_val, texts_val, transform=None)
    test_ds = EmojiDataset(x_test, texts_test, transform=None)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )


    return train_dl, val_dl, test_dl, x_unshuffled, texts_unshuffled
