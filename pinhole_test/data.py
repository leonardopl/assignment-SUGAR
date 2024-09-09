import torch
from torchvision.transforms import v2
import torch.utils.data as tData
import datasets as D
from config import *

def get_transforms():
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=IMG_SIZE, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=IMG_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    return train_transforms, val_transforms

def get_data_loaders():
    train_transforms, val_transforms = get_transforms()

    train_ds = D.PinholeDataset(DATA_DIR, "train", train_transforms)
    val_ds = D.PinholeDataset(DATA_DIR, "val", val_transforms)
    test_ds = D.PinholeDataset(DATA_DIR, "test", val_transforms)

    train_loader = tData.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=8,
    )

    val_loader = tData.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=8,
    )

    test_loader = tData.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=8,
    )

    return train_loader, val_loader, test_loader
