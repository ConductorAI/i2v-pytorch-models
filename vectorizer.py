from pydantic import BaseModel
from image2vec_vit import Img2VecViT
import base64, os

class VectorImagePayload(BaseModel):
  id: str
  image: str

class ImageVectorizer:
  img2vec: Img2VecViT

  def __init__(self, cuda_support, cuda_core):
    self.img2vec = Img2VecViT(cuda_support, cuda_core)

  def vectorize(self, id: str, image: str):
    try:
      filepath = self.saveImage(id, image)
      return self.img2vec.get_vec(filepath)
    except (RuntimeError, TypeError, NameError, Exception) as e:
      print('vectorize error:', e)
      raise e
    finally:
      self.removeFile(filepath)

  def saveImage(self, id: str, image: str):
    try:
      filepath = id
      file_content = base64.b64decode(image)
      with open(filepath, "wb") as f:
        f.write(file_content)
      return filepath
    except Exception as e:
      print(str(e))
      return ""

  def removeFile(self, filepath: str):
    if os.path.exists(filepath):
      os.remove(filepath)
