import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")

import os
from Python.dataset.dataset import CanineLesionsDataset

def get_datasets(image_dir, label_dir, classes, train_csv_path, validation_csv_path, test_csv_path, preprocessing_transforms, augmentation_transforms, patch_based=False):

    train_dataset = CanineLesionsDataset(
        csv_path = train_csv_path,
        image_dir = image_dir,
        label_dir = label_dir,
        classes = classes,
        preprocessing_transforms = preprocessing_transforms,
        augmentation_transforms = augmentation_transforms,
        train = True,
        patch_based = patch_based
    )

    validation_dataset = CanineLesionsDataset(
        csv_path = validation_csv_path,
        image_dir = image_dir,
        label_dir = label_dir,
        classes = classes,
        preprocessing_transforms = preprocessing_transforms,
        augmentation_transforms = None,
        train = False,
        patch_based = patch_based
    )

    # test_dataset = CanineLesionsDataset(
    #     csv_path = test_csv_path,
    #     image_dir = image_dir,
    #     label_dir = label_dir,
    #     classes = classes,
    #     preprocessing_transforms = preprocessing_transforms,
    #     augmentation_transforms = None,
    #     train = False,
    #     patch_based = patch_based
    # )

    return train_dataset, validation_dataset






