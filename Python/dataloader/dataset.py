import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")
import os
import SimpleITK as sitk
import json
import torch.nn
import numpy as np
import torchio as tio

from torch.utils.data import Dataset


class CanineLesionsDataset(Dataset):
    def __init__(self, image_dir, seg_dir, json_dir, mask_dir, classes, spatial_transforms=None, transforms=None):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.mask_dir = mask_dir
        self.json_dir = json_dir
        self.img_names = os.listdir(self.image_dir)
        self.seg_names = os.listdir(self.seg_dir)
        self.mask_names = os.listdir(self.mask_dir)
        self.classes = classes
        self.spatial_transforms = spatial_transforms
        self.transforms = transforms
        self.target_transforms = tio.Compose([
            tio.OneOf(spatial_transforms, p=0.5),
            tio.OneOf(self.transforms, p=0.5)
        ])

        self.reader = sitk.ImageFileReader()
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        self.reader.SetImageIO("MetaImageIO")
        self.reader.SetFileName(os.path.join(self.image_dir, self.img_names[idx]))
        image = self.reader.Execute()
        image_tio = tio.ScalarImage.from_sitk(image)
        tr = tio.transforms.ToCanonical()
        image_tio_r = tr(image_tio)

        self.reader.SetImageIO("NrrdImageIO")
        
        channels = []
        for i, mask in enumerate(os.listdir(os.path.join(self.mask_dir, self.mask_names[idx]))):
            self.reader.SetFileName(os.path.join(self.mask_dir, self.mask_names[idx], mask))
            channel = np.transpose(sitk.GetArrayFromImage(self.reader.Execute()), (1, 2, 0))
            channels.append(channel)

        seg_tio = tio.LabelMap(tensor=torch.ShortTensor(np.array(channels)))

        subject = tio.Subject(
            image = image_tio_r,
            label = seg_tio
        )

        subject_transformed = self.target_transforms(subject)

        return subject_transformed



