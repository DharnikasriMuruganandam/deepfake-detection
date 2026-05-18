from PIL import Image
import os

folder = "dataset/train/fake"

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    try:
        img = Image.open(path)
        img.verify()
    except:
        print("Deleting:", path)
        os.remove(path)