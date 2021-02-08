from tensorflow.keras.datasets import mnist
import skimage.io as io
from tqdm import tqdm
import os

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

label_id = {label: 0 for label in range(10)}
for label in label_id:
    os.makedirs(f"./mnist/{label}", exist_ok=True)

for image, label in tqdm(zip(train_images, train_labels)):
    io.imsave(f"mnist/{label}/{label}_{label_id[label]}.png", image)
    label_id[label] += 1

for image, label in tqdm(zip(test_images, test_labels)):
    io.imsave(f"mnist/{label}/{label}_{label_id[label]}.png", image)
    label_id[label] += 1
