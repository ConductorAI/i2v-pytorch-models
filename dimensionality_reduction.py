import logging
import os
import subprocess

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

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


def train_autoencoder(
    model,
    train_loader,
    eval_loader,
    epochs=20,
    learning_rate=1e-3,
    early_stopping_patience=3,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize variables for Early Stopping
    best_eval_loss = float("inf")
    epochs_no_improve = 0

    logger.info("Training started")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs in train_loader:
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
            # Accumulate the loss
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Evaluation phase
        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for inputs in eval_loader:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(eval_loader)

        logging.info(
            f"Epoch {epoch+1}, Training Loss: {avg_train_loss}, Evaluation Loss: {avg_eval_loss}"
        )

        # Early Stopping check
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    logging.info("Training complete")


def pca_reduce_dimensionality(data_loader, n_components=2000):
    logger.info("PCA started")

    # Concatenate all data samples to fit PCA
    all_data = torch.cat([batch for batch in data_loader], dim=0)

    # Compute PCA
    _, _, V = torch.pca_lowrank(all_data, q=n_components)

    # Display loss
    reduced_data = torch.matmul(all_data, V)
    reconstructed_data = torch.matmul(reduced_data, V.T)
    mse_loss = torch.nn.functional.mse_loss(reconstructed_data, all_data)
    logger.info(f"MSE Loss train: {mse_loss.item()}")

    # Return transformation matrix
    return V


# Downloads parsed images from S3 bucket
def download_images_from_s3(bucket_name: str, local_dir: str):
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


# Split the dataset into training and evaluation sets
def load_dataset(img_dir: str, split_ratio: float):
    dataset = ImageVectorDataset(img_dir=img_dir)
    # Calculate the split sizes
    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    eval_size = total_size - train_size
    # Split dataset into train and validation
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    return train_dataset, eval_dataset


if __name__ == "__main__":
    # Check if local_dir exists and has files in it
    if not os.path.exists(LOCAL_DIR) or not os.listdir(LOCAL_DIR):
        download_images_from_s3(BUCKET_NAME, LOCAL_DIR)
    else:
        logger.info(f"{LOCAL_DIR} already exists and is not empty.")

    # Load the dataset
    train_dataset, eval_dataset = load_dataset(LOCAL_DIR, split_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=True)

    # Initialize
    autoencoder = Autoencoder()

    # Train
    train_autoencoder(autoencoder, train_loader, eval_loader)

    # transform = pca_reduce_dimensionality(train_loader)

    # Save
    model_path = "./autoencoder.pth"
    torch.save(autoencoder.encoder.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # model_path = "./pca_transform_matrix_2048_to_2000.pt"
    # torch.save(transform, model_path)
    # logger.info(f"Transformation matrix saved to {model_path}")
