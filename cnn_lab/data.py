from dataclasses import dataclass
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ANIMAL_IDS = [2, 3, 4, 5, 6, 7]
ANIMAL_NAMES = ["Pájaro", "Gato", "Ciervo", "Perro", "Rana", "Caballo"]

@dataclass
class Data:
    images: torch.Tensor
    labels: torch.Tensor
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    mean: float | torch.Tensor
    std: float | torch.Tensor
    names: list[str]
    source_indices: list[tuple[str, int]] | None = None

    def normalized(self, indices):
        return normalize_images(self.images[indices], self.mean, self.std)


def normalize_images(images, mean, std):
    return (images - mean) / std


def display_image(image):
    """CHW tensor to grayscale HW or RGB HWC for display, without color loss."""
    image = image.detach().cpu() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    return image[0].numpy() if image.shape[0] == 1 else image.permute(1, 2, 0).numpy()


def subset(indices, labels, count, seed):
    if count >= len(indices):
        return indices
    chosen, _ = train_test_split(indices, train_size=count, stratify=labels[indices], random_state=seed)
    return chosen


def prepare(config):
    sources = None
    if config.dataset == "Digits":
        raw = load_digits()
        images = raw.images.astype("float32") / 16
        labels = raw.target
        names = list(map(str, range(10)))
        indices = subset(np.arange(len(labels)), labels, config.samples, config.seed)
        train, rest = train_test_split(indices, test_size=.3, stratify=labels[indices], random_state=config.seed)
        val, test = train_test_split(rest, test_size=.5, stratify=labels[rest], random_state=config.seed)
    else:
        from torchvision import datasets
        animals = config.dataset == "CIFAR10Animals"
        factory = datasets.CIFAR10 if animals else getattr(datasets, config.dataset)
        a = factory("data", train=True, download=True)
        b = factory("data", train=False, download=True)
        a_labels, b_labels = np.asarray(a.targets), np.asarray(b.targets)
        a_indices, b_indices = np.arange(len(a_labels)), np.arange(len(b_labels))
        if animals:
            # Filter before all sampling/splitting; vehicles never enter any split.
            a_indices = a_indices[np.isin(a_labels, ANIMAL_IDS)]
            b_indices = b_indices[np.isin(b_labels, ANIMAL_IDS)]
        count = min(config.samples, len(a_indices))
        train_idx = subset(a_indices, a_labels, count, config.seed)
        test_count = min(max(200, count // 5), len(b_indices))
        test_idx = subset(b_indices, b_labels, test_count, config.seed)
        images = np.concatenate([np.asarray(a.data)[train_idx], np.asarray(b.data)[test_idx]]).astype("float32") / 255
        labels = np.concatenate([a_labels[train_idx], b_labels[test_idx]])
        names = ANIMAL_NAMES.copy() if animals else list(a.classes)
        if animals:
            remap = {original: target for target, original in enumerate(ANIMAL_IDS)}
            labels = np.array([remap[int(label)] for label in labels])
        train, val = train_test_split(np.arange(count), test_size=.2, stratify=labels[:count], random_state=config.seed)
        test = np.arange(count, len(labels))
        sources = [("train", int(i)) for i in train_idx] + [("test", int(i)) for i in test_idx]
    x = torch.tensor(images)
    x = x.permute(0, 3, 1, 2).contiguous() if x.ndim == 4 else x.unsqueeze(1)
    if x.shape[1] == 1:
        # Retain scalar metadata compatibility for older grayscale checkpoints.
        mean = float(x[train].mean()) if config.normalize else 0.
        std = max(float(x[train].std()), 1e-6) if config.normalize else 1.
    else:
        mean = x[train].mean((0, 2, 3), keepdim=True) if config.normalize else torch.zeros(1, x.shape[1], 1, 1)
        std = x[train].std((0, 2, 3), keepdim=True).clamp_min(1e-6) if config.normalize else torch.ones(1, x.shape[1], 1, 1)
    return Data(x, torch.tensor(labels, dtype=torch.long), train, val, test, mean, std, names, sources)


def augment(x, generator=None):
    # Small affine perturbations; never mirror digits. Work on raw [0,1] pixels.
    n = len(x)
    angle = (torch.rand(n, generator=generator) - .5) * .26
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = angle.cos(); theta[:, 1, 1] = angle.cos()
    theta[:, 0, 1] = -angle.sin(); theta[:, 1, 0] = angle.sin()
    theta[:, :, 2] = (torch.rand(n, 2, generator=generator) - .5) * .15
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
