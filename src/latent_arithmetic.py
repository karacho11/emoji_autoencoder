import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List

from .model import ConvAutoencoder



def find_idx_with(substring, texts):
    """
    Returns: all indices whose text descriptions contain the substring.
    """
    substring = substring.lower()
    lst = [i for i, t in enumerate(texts) if substring in t.lower()]
    if not lst:
        raise ValueError(f"No texts contain substring: {substring!r}")
    return lst[0]


@torch.no_grad()
def encode_by_index(
        model: ConvAutoencoder,
        x_unshuffled: torch.Tensor,
        idx: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Returns: latent vectors for the images at given indices
    """

    model.eval().to(device)

    images = x_unshuffled[idx].unsqueeze(0).to(device)        # (N, 3, 64, 64)
    z = model.encode(images)                                  # (N, latent_dim)
    
    return z


@torch.no_grad()
def generate_new_image(
    model:ConvAutoencoder,
    x_unshuffled: torch.Tensor,
    indices: List[int], # [idx_w_attr, idx_wo_attr, idx_base]
    alpha: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Given indices [idx_w_attr, idx_wo_attr, idx_base], compute:

        attr_vector = z_with - z_wo
        z_new  = z_base + alpha * attr_vector

    and decode into a new image.

    Returns: a generated image
    """

    idx_w_attr, idx_wo_attr, idx_base = indices

    z_with = encode_by_index(model, x_unshuffled, idx_w_attr, device=device)
    z_wo = encode_by_index(model, x_unshuffled, idx_wo_attr, device=device)
    z_base = encode_by_index(model, x_unshuffled, idx_base, device=device)

    attr_vector = z_with - z_wo

    z_new = z_base + alpha * attr_vector
    gen_img = model.decode(z_new)

    return gen_img


