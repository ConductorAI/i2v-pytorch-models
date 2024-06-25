import threading

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from autoencoder import Autoencoder


class Img2VecPytorch(object):
    def __init__(self, cuda_support, cuda_core, autoencoder_enabled=False):
        self.device = torch.device(cuda_core if cuda_support else "cpu")

        self.model = models.resnet50(pretrained=True)
        self.layer_output_size = 2048
        self.extraction_layer = self.model._modules.get("avgpool")

        self.model = self.model.to(self.device)

        self.model.eval()

        # Load the autoencoder model
        self.autoencoder_enabled = autoencoder_enabled

        if autoencoder_enabled:
            self.autoencoder = Autoencoder().to(self.device)
            self.autoencoder.encoder.load_state_dict(torch.load("./autoencoder.pth"))
            self.autoencoder.eval()
            # self.transform_matrix = torch.load("pca_transform_matrix_2048_to_2000.pt")

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()
        self.lock = threading.Lock()

    def get_vec(self, image_path):
        img = Image.open(image_path).convert("RGB")

        with self.lock:
            image = (
                self.normalize(self.to_tensor(self.scaler(img)))
                .unsqueeze(0)
                .to(self.device)
            )
            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1, device=self.device)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            self.model(image)
            h.remove()

            if self.autoencoder_enabled:
                # Dimensionality reduction via Autoencoder
                reduced_embedding = self.autoencoder.encoder(
                    my_embedding.view(my_embedding.size(0), -1)
                )

                # Dimensionality reduction via PCA
                # reduced_embedding = torch.matmul(
                #     my_embedding.view(my_embedding.size(0), -1), self.transform_matrix
                # )

                return reduced_embedding.detach().cpu().squeeze().numpy()
            else:
                return my_embedding.detach().cpu().squeeze().numpy()
