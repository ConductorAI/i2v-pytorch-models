import logging
import os
import subprocess

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from autoencoder import Autoencoder
from image_vector_dataset import ImageVectorDataset

BUCKET_NAME = "test"
LOCAL_DIR = "./s3_images"
S3_ENDPOINT = "http://localhost:4566"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def train_autoencoder(model, data_loader, epochs=5, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    logger.info("Training started")

    for epoch in range(epochs):
        for inputs in data_loader:
            # Zero out the gradients
            optimizer.zero_grad()
            # Get output from the Autoencoder model
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, inputs)
            # Compute the gradients via backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()

        logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

    logging.info("Training complete")


# Downloads parsed images from S3 bucket
def download_images_from_s3(bucket_name, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    cmd = f"aws --endpoint-url=http://{S3_ENDPOINT} s3 ls s3://{bucket_name}/"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()

    for line in lines:
        file_name = line.split()[-1]
        # Filter out PDF files which are stored in the same S3 bucket as images
        if not file_name.lower().endswith((".pdf")):
            download_cmd = f"aws --endpoint-url={S3_ENDPOINT} s3 cp s3://{bucket_name}/{file_name} {local_dir}/{file_name}"
            subprocess.run(download_cmd, shell=True)
    logger.info("Download complete")


if __name__ == "__main__":
    # Check if local_dir exists and has files in it
    if not os.path.exists(LOCAL_DIR) or not os.listdir(LOCAL_DIR):
        download_images_from_s3(BUCKET_NAME, LOCAL_DIR)
    else:
        logger.info(f"{LOCAL_DIR} already exists and is not empty.")

    dataset = ImageVectorDataset(img_dir=LOCAL_DIR)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize
    autoencoder = Autoencoder()

    # Train
    train_autoencoder(autoencoder, data_loader)

    # Save
    model_path = "./autoencoder.pth"
    torch.save(autoencoder.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
