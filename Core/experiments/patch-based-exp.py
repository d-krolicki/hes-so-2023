import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")

import torch
import torchio as tio
import os
import time
import wandb
import logging

from monai.losses import DiceLoss
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from torch.utils.data import DataLoader

from Core.datasets import get_datasets

DATA_ROOT = "/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz/Data/COMPLETE_DATA/all_classes/unit_spacing"

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
        tio.Clamp(out_min = -2000, out_max=4000),
        tio.RescaleIntensity((0, 1))
    ])

AUGMENTATION_TRANSFORMS = tio.Compose([
        tio.RandomFlip(
            axes = (0,1,2),
            p=0.8),
        tio.RandomAffine(),
        tio.RandomGamma(
            log_gamma = (0.3,0.3),
            p=0.75),
        tio.RandomNoise(
            mean = 0,
            std = (0, 0.25),
            p=0.75)
    ])

BATCH_SIZE = 16

PATCH_SIZE = 128
QUEUE_LENGTH = 50
SAMPLES_PER_VOLUME = 16

NUM_EPOCHS = 100

LR = 1e-3

MODEL_NAME = "patch-based-29-08-2023-001"

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
print("Loading datasets")

train, validation = get_datasets(
    image_dir = IMAGE_DIR,
    label_dir = LABEL_DIR,
    classes = CLASSES,
    train_csv_path=os.path.join(CSV_DIR, "train_patch_based.csv"),
    validation_csv_path=os.path.join(CSV_DIR, "validation_patch_based.csv"),
    test_csv_path=os.path.join(CSV_DIR, "test_patch_based.csv"),
    preprocessing_transforms = PREPROCESSING_TRANSFORMS,
    augmentation_transforms = AUGMENTATION_TRANSFORMS,
    patch_based=True
)
print("Datasets loaded.")

sampler = tio.data.UniformSampler(patch_size=PATCH_SIZE)

patches_train_queue = tio.Queue(
    subjects_dataset = train.get_subjects_dataset(),
    max_length = QUEUE_LENGTH,
    samples_per_volume = SAMPLES_PER_VOLUME,
    sampler = sampler,
    num_workers = 8
)

patches_train_loader = DataLoader(
    patches_train_queue,
    batch_size = BATCH_SIZE,
    num_workers = 0
)

patches_validation_queue = tio.Queue(
    subjects_dataset = validation.get_subjects_dataset(),
    max_length = QUEUE_LENGTH,
    samples_per_volume = SAMPLES_PER_VOLUME,
    sampler = sampler,
    num_workers = 8
)

patches_validation_loader = DataLoader(
    patches_validation_queue,
    batch_size = BATCH_SIZE,
    num_workers = 0
)

model = Unet(
    spatial_dims = 3,
    in_channels = 1,
    out_channels = len(CLASSES),
    channels = (8, 16, 32, 64, 128, 256),
    strides = (2,2,2,2,2),
    num_res_units = 4,
    norm = Norm.INSTANCE
).to(device)

loss_fcn = DiceLoss(sigmoid=True)
opt = torch.optim.Adam(model.parameters() , LR)

run = wandb.init(
    project="patch-based experiments",
)
wandb.config = {
    "epochs" : NUM_EPOCHS,
    "learning_rate" : LR,
    "batch_size" : BATCH_SIZE,
    "patch_size" : PATCH_SIZE,
    "patches_per_image" : SAMPLES_PER_VOLUME
}

class NullWriter(object):
    def write(self, arg):
        pass

nullwrite = NullWriter()
oldstdout = sys.stdout

logger = logging.getLogger("Train")

print("Beginning training")
for epoch in range(NUM_EPOCHS):
    t1 = time.time()

    per_epoch_average_train_loss = 0
    per_epoch_average_validation_loss = 0
    model.train()

    l = len(patches_train_loader)
    for i, patches_batch in enumerate(patches_train_loader):
        tin = time.time()
        
        opt.zero_grad()

        inputs = patches_batch['image'][tio.DATA]
        labels = torch.cat([patches_batch[mask][tio.DATA] for mask in list(patches_batch.keys())[1:-1]], dim=1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)

        loss = loss_fcn(outputs, labels)

        per_epoch_average_train_loss += loss

        loss.backward()

        opt.step()
        tout = time.time()
        sys.stdout.write(f"\rImage {i+1} / {l}, time elapsed: {tout-tin:.2f}")
        sys.stdout.flush()

    model.eval()
    with torch.no_grad():

        l = len(patches_validation_loader)
        for i, patches_batch in enumerate(patches_validation_loader):
            tin = time.time()

            inputs = patches_batch['image'][tio.DATA]
            labels = torch.cat([patches_batch[mask][tio.DATA] for mask in list(patches_batch.keys())[1:-1]], dim=1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)

            loss = loss_fcn(outputs, labels)

            per_epoch_average_train_loss += loss

            tout = time.time()

            sys.stdout.write(f"\rImage {i+1} / {l}, time elapsed: ")
            sys.stdout.flush()

    per_epoch_average_train_loss = per_epoch_average_train_loss / (len(patches_train_loader) * BATCH_SIZE)
    per_epoch_average_validation_loss = per_epoch_average_validation_loss / (len(patches_validation_loader) * BATCH_SIZE)

    if per_epoch_average_train_loss < best_per_epoch_average_train_loss:
        print(f"Average train loss improved from {best_per_epoch_average_train_loss:.4f} to {per_epoch_average_train_loss:.4f}. Saving the model...")
        torch.save(
            obj=model,
            f="/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz/Results/experiments/patch-based-experiments/"+MODEL_NAME+".pth"
        )
        best_per_epoch_average_train_loss = per_epoch_average_train_loss
    else:
        print(f"Loss not improved - model checkpoint was not created.")

    wandb.log({
        "PerEpochAverageTrainLoss" : per_epoch_average_train_loss,
        "PerEpochAverageValidationLoss" : per_epoch_average_validation_loss
    })

    print()
    t2 = time.time()
    print(f"Time elapsed: {int(t2-t1)} seconds.")
        

