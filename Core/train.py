import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")

import torch
import os
import monai
import monai.transforms as mt
import torchio as tio
import ignite
from Python.dataloader.dataset import CanineLesionsDataset
from torch import nn
from torch.utils.data import DataLoader

DATA_SOURCE_FOLDERPATH = "/OLD-DATA-STOR/segmentation ovud/final-data/all-checked"
DATA_WITH_NEUTRAL_SAMPLES_SOURCE_FOLDERPATH = "/OLD-DATA-STOR/segmentation ovud/final-data-with-neutral-samples/all-checked"

device = "cpu"
# (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
print(f"Using {device} device")

dataset = CanineLesionsDataset(
    image_dir = os.path.join(DATA_SOURCE_FOLDERPATH, "images"),
    seg_dir = os.path.join(DATA_SOURCE_FOLDERPATH, "segmentations"),
    json_dir = os.path.join(DATA_SOURCE_FOLDERPATH, "seg-json"),
    mask_dir = os.path.join(DATA_SOURCE_FOLDERPATH, "masks"),
    classes = ['axillary lymph node', 'sternal lymph node', 'hepatic mass',
                'lung consolidation', 'lipoma', 'subcutaneous mass',
                'lung mass', 'interstitial pattern', "unidentified node"],
    spatial_transforms={
        tio.RandomElasticDeformation() : 0.2,
        tio.RandomAffine() : 0.8
    },
    transforms = {
        tio.RescaleIntensity(out_min_max=(0, 1)) : 0.5
    }
)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

model = monai.networks.nets.Unet(
    spatial_dims = 3,
    in_channels = 9,
    out_channels = 9,
    channels = (16, 32, 64, 128, 256),
    strides = (2, 2, 2, 2),
    num_res_units = 2,
    norm = monai.networks.layers.Norm.BATCH
).to(device)

loss_fcn = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
opt = torch.optim.Adam(model.parameters() ,1e-2)

