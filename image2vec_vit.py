import threading

import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

# https://huggingface.co/google/vit-base-patch16-224
MODEL_NAME = "google/vit-base-patch16-224"


class Img2VecViT:
    def __init__(self, cuda_support, cuda_core):
        self.device = torch.device(cuda_core if cuda_support else "cpu")

        self.model = ViTModel.from_pretrained(MODEL_NAME)

        self.layer_output_size = self.model.config.hidden_size

        if self.layer_output_size != 768:
            raise ValueError(
                "Only ViT models with hidden size of 768 are supported at the moment"
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        self.lock = threading.Lock()

    def get_vec(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.lock:
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)

        return features.cpu().numpy()[0]
