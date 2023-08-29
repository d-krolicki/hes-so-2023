import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")

import torch
import torchio as tio
import os
import time

from monai.losses import DiceLoss
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from torch.utils.data import DataLoader

from Python.dataset.dataset import CanineLesionsUniformSampler
from datasets import get_datasets

DATA_ROOT = "/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz/Data"

IMAGE_DIR = os.path.join(DATA_ROOT, "images")
LABEL_DIR = os.path.join(DATA_ROOT, "labels")
CSV_DIR = os.path.join(DATA_ROOT, "csv")

ORIGINAL_CLASSES = ['axillary lymph node',
           'sternal lymph node',
           'hepatic mass',
           'lung consolidation',
           'lipoma',
           'subcutaneous mass',
           'lung mass',
           'interstitial pattern']

CUSTOM_CLASSES = [
    'unidentified node'
]

CLASSES = ORIGINAL_CLASSES + CUSTOM_CLASSES

PREPROCESSING_TRANSFORMS = tio.Compose([
        tio.RescaleIntensity((0, 1))
    ])

AUGMENTATION_TRANSFORMS = tio.Compose([
        tio.RandomAffine(),
        tio.RandomGamma(p=0.5),
        tio.RandomNoise(p=0.5),
        tio.RandomMotion(degrees=10, translation=10, p=0.25),
        tio.RandomBiasField(p=0.25),
        tio.RandomFlip(p=0.25),
    ])

BATCH_SIZE = 1

PATCH_SIZE = (128,128,128)
NUM_PATCHES = 8

NUM_EPOCHS = 10

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

train, validation, test = get_datasets(
    csv_dir = CSV_DIR,
    image_dir = IMAGE_DIR,
    label_dir = LABEL_DIR,
    classes = CLASSES,
    preprocessing_transforms = PREPROCESSING_TRANSFORMS,
    augmentation_transforms = AUGMENTATION_TRANSFORMS,
    dummy_train_data=True
)

train_dataloader = DataLoader(
    dataset = train, 
    batch_size = BATCH_SIZE, 
    shuffle = False)
validation_dataloader = DataLoader(
    dataset = validation, 
    batch_size = BATCH_SIZE, 
    shuffle = False)
test_dataloader = DataLoader(
    dataset = validation, 
    batch_size = BATCH_SIZE, 
    shuffle = False)

sampler = CanineLesionsUniformSampler(
    patch_size = PATCH_SIZE
    )

model = Unet(
    spatial_dims = 3,
    in_channels = 1,
    out_channels = len(CLASSES),
    channels = (16, 32, 64, 128, 256),
    strides = (2,2,2,2),
    num_res_units = 2,
    norm = Norm.INSTANCE
).to(device)

loss_fcn = DiceLoss(sigmoid=True)
opt = torch.optim.Adam(model.parameters() ,1e-2)

for epoch in range(NUM_EPOCHS):
    t1 = time.time()
    print(f"Epoch {epoch}")

    for _ in range(len(train_dataloader)):
        subject = tio.utils.get_subjects_from_batch(next(iter(train_dataloader)))[0]

        for i in range(NUM_PATCHES):
            sample = next(iter(sampler(subject)))

            inputs, labels = torch.unsqueeze(sample[0], dim=0).to(device), torch.unsqueeze(torch.cat(sample[1:]), dim=0).to(device)

            if epoch > 0:
                opt.zero_grad()

            outputs = model(inputs)

            loss = loss_fcn(outputs, labels)
            print(f"Dice loss: {loss}")

        loss.backward()

        opt.step()
        t2 = time.time()
    print(f"Time elapsed: {t2-t1:.2f} seconds.")

# basic

# for epoch in range(NUM_EPOCHS):
#     t1 = time.time()
#     print(f"Epoch {epoch}")
    
#     for _ in range(len(train_dataloader)):
#         subject = tio.utils.get_subjects_from_batch(next(iter(train_dataloader)))[0]

#         inputs, labels = torch.unsqueeze(subject[0], dim=0).to(device), torch.unsqueeze(torch.cat(subject[1:]), dim=0).to(device)

#         if epoch > 0:
#             opt.zero_grad()

#         outputs = model(inputs)

#         for i in range(len(CLASSES)):
#             loss = loss_fcn(outputs, labels[0,i,:,:,:])
#             print(f"Dice loss: {loss}")

#         loss.backward()

#         opt.step()
#     t2 = time.time()
#     print(f"Time elapsed: {t2-t1:.2f} seconds.")

# patch-based






































# train_dataset = CanineLesionsDataset(
#     csv_path = os.path.join(CSV_DIR, "train.csv"),
#     image_dir = IMAGE_DIR,
#     label_dir = LABEL_DIR,
#     classes = CLASSES,
#     preprocessing_transforms = tio.Compose([
#         tio.RescaleIntensity((0, 1)),
#         tio.OneHot()
#     ]),
#     augmentation_transforms=tio.Compose([
#         tio.RandomAffine(),
#         tio.RandomGamma(p=0.5),
#         tio.RandomNoise(p=0.5),
#         tio.RandomMotion(degrees=10, translation=10, p=0.25),
#         tio.RandomBiasField(p=0.25),
#         tio.RandomFlip(p=0.25),
#     ]),
#     train=True
# )

# validation_dataset = CanineLesionsDataset(
#     csv_path = os.path.join(CSV_DIR, "validation.csv"),
#     image_dir = IMAGE_DIR,
#     label_dir = LABEL_DIR,
#     classes = CLASSES,
#     preprocessing_transforms = tio.Compose([
#         tio.RescaleIntensity((0, 1))
#     ]),
#     augmentation_transforms=tio.Compose([
#         tio.RandomAffine(),
#         tio.RandomGamma(p=0.5),
#         tio.RandomNoise(p=0.5),
#         tio.RandomMotion(degrees=10, translation=10, p=0.25),
#         tio.RandomBiasField(p=0.25),
#         tio.RandomFlip(p=0.25),
#     ]),
#     train=False
# )

# test_dataset = CanineLesionsDataset(
#     csv_path = os.path.join(CSV_DIR, "test.csv"),
#     image_dir = IMAGE_DIR,
#     label_dir = LABEL_DIR,
#     classes = CLASSES,
#     preprocessing_transforms = tio.Compose([
#         tio.RescaleIntensity((0, 1))
#     ]),
#     augmentation_transforms=tio.Compose([
#         tio.RandomAffine(),
#         tio.RandomGamma(p=0.5),
#         tio.RandomNoise(p=0.5),
#         tio.RandomMotion(degrees=10, translation=10, p=0.25),
#         tio.RandomBiasField(p=0.25),
#         tio.RandomFlip(p=0.25),
#     ]),
#     train=False
# )