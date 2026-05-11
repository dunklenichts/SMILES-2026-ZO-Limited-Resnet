"""
head_init.py — Semantic initialization for the CIFAR100 head.

Instead of using only random initialization, this solution uses 
the pretrained ImageNet classifier head from ResNet18. 
"""

import os
import re
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


_SEARCH_TERMS = {
    "aquarium_fish": ["goldfish", "fish"],
    "cattle": ["ox", "cow", "bull"],
    "couch": ["sofa", "couch"],
    "cup": ["cup", "mug"],
    "dinosaur": ["triceratops"],
    "flatfish": ["flatfish", "stingray", "electric ray"],
    "lawn_mower": ["lawn mower"],
    "maple_tree": ["maple"],
    "oak_tree": ["acorn"],
    "palm_tree": ["palm"],
    "pickup_truck": ["pickup", "truck"],
    "pine_tree": ["pine"],
    "ray": ["stingray", "electric ray"],
    "rocket": ["missile", "projectile"],
    "sea": ["seashore", "coral reef", "lakeside"],
    "seal": ["seal", "sea lion"],
    "streetcar": ["streetcar", "trolleybus"],
    "sweet_pepper": ["bell pepper"],
    "telephone": ["telephone", "cellular telephone", "pay-phone"],
    "television": ["television", "monitor", "screen"],
    "train": ["train", "locomotive"],
    "willow_tree": ["willow"],
}


def _load_cifar100_classes() -> list[str]:
    """Load CIFAR100 fine class names from the local dataset metadata.
    
    Returns:
        A list of CIFAR100 fine class names in the official dataset order.
    """

    possible_roots = [
        os.environ.get("CIFAR100_ROOT"),
        "./data",
        "data",
    ]

    for root in possible_roots:
        if root is None:
            continue

        meta_path = Path(root) / "cifar-100-python" / "meta"

        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f, encoding="latin1")
            return meta["fine_label_names"]

    for meta_path in Path(".").glob("**/cifar-100-python/meta"):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        return meta["fine_label_names"]

    raise FileNotFoundError(
        "Could not find CIFAR100 metadata file. "
        "Expected something like ./data/cifar-100-python/meta"
    )


def _tokens(text: str) -> set[str]:
    """Convert a class name into a set of simple lowercase words

    Args:
        text: A class name or search phrase.

    Returns:
        A set of lowercase words.
    """
    text = text.lower().replace("_", " ")
    return set(re.findall(r"[a-z]+", text))


def _find_matches(categories: list[str], cifar_name: str) -> list[int]:
    """Find ImageNet classes that are related to one CIFAR100 class.
    
    Args:
        categories: List of ImageNet class names
        cifar_name: Name of one CIFAR100 class

    Returns:
        A list of matching ImageNet class indices
    """

    search_terms = _SEARCH_TERMS.get(
        cifar_name,
        [cifar_name.replace("_", " ")]
    )

    matches = set()

    for term in search_terms:
        term_tokens = _tokens(term)

        for idx, category in enumerate(categories):
            category_tokens = _tokens(category)

            if term_tokens.issubset(category_tokens):
                matches.add(idx)

    return sorted(matches)


def init_last_layer(layer: nn.Linear) -> None:

    semantic_scale = 0.75

    with torch.no_grad():
        nn.init.xavier_uniform_(layer.weight)
        layer.weight.data.mul_(0.01)
        nn.init.zeros_(layer.bias)

        cifar_classes = _load_cifar100_classes()

        weights_enum = models.ResNet18_Weights.IMAGENET1K_V1
        imagenet_model = models.resnet18(weights=weights_enum)

        imagenet_w = imagenet_model.fc.weight.detach()
        categories = weights_enum.meta["categories"]

        matched_count = 0

        for cifar_idx, cifar_name in enumerate(cifar_classes):
            match_indices = _find_matches(categories, cifar_name)

            if not match_indices:
                continue

            semantic_weight = imagenet_w[match_indices].mean(dim=0)
            layer.weight[cifar_idx].copy_(semantic_scale * semantic_weight)

            matched_count += 1

        print(f"[head_init] semantic ImageNet matches: {matched_count}/100")