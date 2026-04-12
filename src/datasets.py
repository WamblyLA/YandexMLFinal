from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, dataframe, transform=None, return_path=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = row["target"]
        path = row["path"]

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, torch.tensor(label, dtype=torch.float32), str(path)

        return image, torch.tensor(label, dtype=torch.float32)
    
class FaceTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.files = sorted(self.image_dir.iterdir(), key=lambda x: int(x.stem))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert("RGB")
        image_id = int(path.stem)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id
    
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])