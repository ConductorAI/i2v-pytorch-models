from transformers import ViTModel, ViTImageProcessor
from image2vec_vit import MODEL_NAME


model = ViTModel.from_pretrained(MODEL_NAME)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

print(f"Model and processor for {MODEL_NAME} downloaded successfully.")
