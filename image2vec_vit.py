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
        # If one of the image dimensions is 1 or 3 it can confuse the infer_channel_dimension_format function
        # in the Transformers library, so we set the input_data_format to "channels_last" in that case
        input_data_format = (
            "channels_last" if img.width in (1, 3) or img.height in (1, 3) else None
        )
        try:
            inputs = self.processor(
                images=img, return_tensors="pt", input_data_format=input_data_format
            )
        except ValueError:
            # The conversion of a PIL.Image.Image to a numpy array should yield the following shape:
            # (width, height, channels) where channels is the last dimension
            # On the off chance this isn't the case, we try again with the input_data_format set to "channels_first"
            inputs = self.processor(
                images=img, return_tensors="pt", input_data_format="channels_first"
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.lock:
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)

        return features.cpu().numpy()[0]
