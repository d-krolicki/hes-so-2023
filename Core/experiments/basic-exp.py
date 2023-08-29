import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")

import torch
import torchio as tio
import os
import time
import wandb

from monai.losses import DiceLoss
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from torch.utils.data import DataLoader

from Core.datasets import get_datasets

DATA_ROOT = "/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz/Data/COMPLETE_DATA/all_classes/unified_resolution"

IMAGE_DIR = os.path.join(DATA_ROOT, "images")
LABEL_DIR = os.path.join(DATA_ROOT, "labels")
CSV_DIR = os.path.join(DATA_ROOT, "csv")

TRAIN_CSV_PATH = os.path.join(CSV_DIR, "train.csv")
VALIDATION_CSV_PATH = os.path.join(CSV_DIR, "validation.csv")
TEST_CSV_PATH = os.path.join(CSV_DIR, "test.csv")

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
        tio.Clamp(out_min = -2000, out_max = 4000),
        tio.RescaleIntensity(out_min_max=(0, 1))
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

MODEL_NAME = "29-08-BASIC-001"
PRETRAINED_MODEL_NAME = "24-08-BASIC-001"

BATCH_SIZE = 1
LR = 1e-3

NUM_EPOCHS = 100

device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

train, validation = get_datasets(
    image_dir = IMAGE_DIR,
    label_dir = LABEL_DIR,
    classes = CLASSES,
    train_csv_path = TRAIN_CSV_PATH,
    validation_csv_path = VALIDATION_CSV_PATH,
    test_csv_path = TEST_CSV_PATH,
    preprocessing_transforms = PREPROCESSING_TRANSFORMS,
    augmentation_transforms = AUGMENTATION_TRANSFORMS,
    patch_based=False
)

train_dataloader = DataLoader(
    dataset = train, 
    batch_size = BATCH_SIZE, 
    shuffle = True,
    num_workers=0)
validation_dataloader = DataLoader(
    dataset = validation, 
    batch_size = BATCH_SIZE, 
    shuffle = False,
    num_workers=0)

model = Unet(
    spatial_dims = 3,
    in_channels = 1,
    out_channels = len(CLASSES),
    channels = (8, 16, 32, 64, 128, 256),
    strides = (2,2,2,2,2),
    num_res_units = 4,
    norm = (Norm.LOCALRESPONSE, {"size" : 2})
).to(device)

# model = torch.load("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz/Results/experiments/basic-experiments/"+PRETRAINED_MODEL_NAME+".pth").to(device)
# print(f"Loading the model: {PRETRAINED_MODEL_NAME}")

loss_fcn = DiceLoss(sigmoid=True)
opt = torch.optim.Adam(model.parameters(), LR)

run = wandb.init(
    project="basic experiment",
)
wandb.config = {
    "epochs" : NUM_EPOCHS,
    "learning_rate" : LR,
    "batch_size" : BATCH_SIZE,
    "preprocessing-transforms" : PREPROCESSING_TRANSFORMS,
    "augmentation-transforms" : AUGMENTATION_TRANSFORMS,
    "model_type" : "ResidualUNet",
    "model" : MODEL_NAME
}

best_per_epoch_average_train_loss = 1

for epoch in range(NUM_EPOCHS):
    print("Training")
    t1 = time.time()
    per_epoch_average_train_loss = 0
    per_epoch_average_validation_loss = 0
    sys.stdout.write(f"\rEpoch {epoch+1} / {NUM_EPOCHS}")
    sys.stdout.flush()
    print()
    model.train()
    l = len(train_dataloader)
    for i, data in enumerate(train_dataloader):
        tin = time.time()
        
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        print(f"train min(inputs):{torch.min(inputs)}")
        print(f"train min(labels):{torch.min(labels)}")
        print(f"train min(outputs):{torch.min(outputs)}")
        print(f"train max(inputs):{torch.max(inputs)}")
        print(f"train max(labels):{torch.max(labels)}")
        print(f"train max(outputs):{torch.max(outputs)}")

        loss = loss_fcn(outputs, labels)

        print(f"train loss:{loss}")

        per_epoch_average_train_loss += loss

        loss.backward()
        opt.step()

        opt.zero_grad()

        tout = time.time()

        sys.stdout.write(f"\rImage {i+1} / {l}, time elapsed: {tout-tin:.2f}\n")
        sys.stdout.flush()
    
    l = len(validation_dataloader)
    print("\nValidation")
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):
            tin = time.time()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            print(f"validation min(inputs):{torch.min(inputs)}")
            print(f"validation min(labels):{torch.min(labels)}")
            print(f"validation min(outputs):{torch.min(outputs)}")
            print(f"validation max(inputs):{torch.max(inputs)}")
            print(f"validation max(labels):{torch.max(labels)}")
            print(f"validation max(outputs):{torch.max(outputs)}")
            loss = loss_fcn(outputs, labels)
            print(f"validation loss: {loss}")
            per_epoch_average_validation_loss += loss
            
            tout = time.time()

            sys.stdout.write(f"\rImage {i+1} / {l}, time elapsed: {tout-tin:.2f} seconds\n")
            sys.stdout.flush()

    per_epoch_average_train_loss = per_epoch_average_train_loss / (len(train_dataloader) * BATCH_SIZE)
    per_epoch_average_validation_loss = per_epoch_average_validation_loss / (len(validation_dataloader) * BATCH_SIZE)

    if per_epoch_average_train_loss < best_per_epoch_average_train_loss:
        print(f"Average train loss improved from {best_per_epoch_average_train_loss:.4f} to {per_epoch_average_train_loss:.4f}. Saving the model...")
        torch.save(
            obj=model,
            f="/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz/Results/experiments/basic-experiments/"+MODEL_NAME+".pth"
        )
        best_per_epoch_average_train_loss = per_epoch_average_train_loss
    else:
        print(f"Loss not improved - model checkpoint was not created.")

    print(f"PerEpochAverageTrainLoss : {per_epoch_average_train_loss}")
    print(f"PerEpochAverageValidationLoss : {per_epoch_average_validation_loss}")
    wandb.log({
        "PerEpochAverageTrainLoss" : per_epoch_average_train_loss,
        "PerEpochAverageValidationLoss" : per_epoch_average_validation_loss
    })

    print()
    t2 = time.time()
    print(f"Time elapsed: {int(t2-t1)} seconds.")






