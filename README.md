# Emoji Style Editor (Convolutional Autoencoder)

**Short summary:**  
A convolutional autoencoder trained on the [valhalla/emoji-dataset] emoji images, with a latent-space editor that can add or remove visual attributes (e.g., glasses, hearts, tears) via simple vector arithmetic. Implemented in PyTorch and packaged with a small interactive web demo (Gradio).

---

## Motivation

Emoji are simple but expressive visual symbols – perfect for experimenting with deep learning on images.  
This project turns a class assignment on emoji autoencoders into a full portfolio project:

- Train a convolutional autoencoder to reconstruct emoji images
- Explore the structure of the latent space
- Learn **attribute vectors** (e.g., “+ glasses”, “+ hearts”, “+ tears”)
- Build a small app where users can pick a base emoji + attribute and generate new emojis

---

## Dataset

- **Name:** `valhalla/emoji-dataset`
- **Source:** HuggingFace Datasets
- ~2.7k emoji images, each with:
  - `image`: PNG emoji rendered as a 256×256 RGB image
  - `text`: short description (e.g., `"smiling face with heart-shaped eyes"`)

For training, images are:

- Resized to 64×64
- Normalized to `[0, 1]`
- Used without any labels (purely unsupervised reconstruction)

---

## Model

### Convolutional Autoencoder

- **Encoder**
  - 4 convolutional layers with stride 2, ReLU
  - Input: `(3, 64, 64)`
  - Bottleneck: `latent_dim`-dimensional vector (e.g., 128)

- **Decoder**
  - 4 transpose-convolution layers with stride 2, ReLU
  - Final layer uses `Sigmoid` to output pixels in `[0, 1]`

- **Loss**
  - Mean Squared Error (MSE) between input and reconstructed image

- **Optimizer**
  - Adam (default LR: `1e-3`)

---

## Training

```bash
# create env, install deps
pip install -r requirements.txt

# run training (default config)
python -m src.train
