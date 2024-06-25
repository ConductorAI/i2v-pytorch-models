from torch import nn

"""
Autoencoder

This model is currently used to reduce the dimensionality of the image vectors produced by the ResNet50 model.
ResNet, in practice, has proven itself to be more effective (confident) at providing high similarity scores for
(near) exact image matches than various ViT models (primarily google/vit-base-patch16-224). However,
google/vit-base-patch16-224 performs much better in distinguishing between images that are similar and images that
are disimilar while also being able to providing more than adequate results for exact matches. Additionally, ResNet
tends to embed images more closely together than ViT models we have tested, which means even visually different images
tend to produce higher similarity scores than intuition says they should.

For this reason, we are continuing the use of google/vit-base-patch16-224 for our image inference but will keep this Autoencoder model
if we need to return to ResNet50 for any reason in the future.

The reason the autoencoder is needed is because vectors indexed within the PGVector database are limited to 2000 dimensions.
Therefore, this simple model has been trained on our parsed image dataset to reduce the dimensionality of the vectors generated
by the ResNet50 model from 2048 to 2000.
"""

VECTORIZER_OUTPUT_DIM = 2048
TARGET_VECTOR_DIM = 2000


class Autoencoder(nn.Module):
    def __init__(self, input_dim=VECTORIZER_OUTPUT_DIM, encoding_dim=TARGET_VECTOR_DIM):
        super(Autoencoder, self).__init__()

        """
        Encoder
        VECTORIZER_OUTPUT_DIM -> TARGET_VECTOR_DIM
        The dimensionality reduction for this use case is so minor that good
        results can be achieved with a single pair of linear and activation layers.
        """
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True),
        )

        """
        Decoder
        TARGET_VECTOR_DIM -> VECTORIZER_OUTPUT_DIM
        """
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
