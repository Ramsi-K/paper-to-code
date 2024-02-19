from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class A2BDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B

        self.transform = transform

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(
            len(self.A_images), len(self.B_images)
        )  # 1000, 1500
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)
        print(f"Total A Images: {self.A_len}")
        print(f"Total B Images: {self.B_len}")

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)

        A_img = np.array(Image.open(A_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))

        if self.transform:
            augmentation_A = self.transform(image=A_img)
            augmentation_B = self.transform(image=B_img)
            A_img = augmentation_A["image"]
            B_img = augmentation_B["image"]

        return A_img, B_img
