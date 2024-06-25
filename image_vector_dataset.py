import logging
import os

import torch
from torch.utils.data import Dataset

from image2vec import Img2VecPytorch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

"""
ImageVectorDataset

This is a PyTorch Dataset class that takes a directory of images and, when accessed,
returns the vector produced by the Img2VecPytorch model for a given image.
"""


class ImageVectorDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = [img for img in os.listdir(img_dir)]

        cuda_support = torch.cuda.is_available()
        if cuda_support:
            cuda_core = "cuda:0"
            logger.info("Running on GPU")
        else:
            cuda_core = ""
            logger.info("Running on CPU")

        self.img2vec = Img2VecPytorch(cuda_support, cuda_core)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        # The img2vec model handles the image loading, transformation, and vectorization
        vector = self.img2vec.get_vec(img_path)
        return vector
