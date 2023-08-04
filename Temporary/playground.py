import numpy as np
import SimpleITK as sitk
import os
import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")
import torchio as tio
from Python.dataloader.dataset import CanineLesionsDataset

DATA_SOURCE_FOLDERPATH = "/OLD-DATA-STOR/segmentation ovud/final-data/all-checked"
DATA_WITH_NEUTRAL_SAMPLES_SOURCE_FOLDERPATH = "/OLD-DATA-STOR/segmentation ovud/final-data-with-neutral-samples/all-checked"

dataset = CanineLesionsDataset(
    image_dir = os.path.join(DATA_SOURCE_FOLDERPATH, "images"),
    seg_dir = os.path.join(DATA_SOURCE_FOLDERPATH, "segmentations"),
    json_dir = os.path.join(DATA_SOURCE_FOLDERPATH, "seg-json"),
    mask_dir = os.path.join(DATA_SOURCE_FOLDERPATH, "masks"),
    classes = ['axillary lymph node', 'sternal lymph node', 'hepatic mass',
                'lung consolidation', 'lipoma', 'subcutaneous mass',
                'lung mass', 'interstitial pattern', "unidentified node"],
    spatial_transforms={
        # tio.RandomElasticDeformation() : 0.2,
        # tio.RandomAffine() : 0.8
        },
    transforms = {
        # tio.RescaleIntensity(out_min_max=(0, 1)) : 0.5
    }
)

for key in dataset[0]:
    print(key)

print(np.unique(dataset[0]['channel5'], return_counts=True))
print(dataset[0]['image'].shape)
print(dataset[0]['image'].orientation)
print(dataset[0]['image'].spacing)
print(dataset[0]['channel5'].shape)
print(dataset[0]['channel5'].orientation)
print(dataset[0]['channel5'].spacing)