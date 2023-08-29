import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")
import os
import torch.nn
import csv
import torchio as tio
import time

from typing import Optional, Generator, List
from torchio.data.subject import Subject
from torch.utils.data import Dataset

# class CanineLesionsSubjectsDataset(tio.SubjectsDataset):
#     def __init__(
#         self,
#         csv_path:str,
#         image_dir:str,
#         label_dir:str,
#         classes:List[str],
#         preprocessing_transforms,
#         augmentation_transforms,
#         train=True
#     ):
#         self.csv_path = csv_path
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.classes = classes
#         self.preprocessing_transforms = preprocessing_transforms
#         self.augmentation_transforms = augmentation_transforms
#         self.train = train

#         self.subjects = []

#         with open(csv_path, "r", newline='') as f:
#             self.reader = csv.reader(f, delimiter="#")
#             for row in self.reader:
#                 filename = row[:-4]

#                 subject = tio.Subject()

                

class CanineLesionsDataset(Dataset):
    def __init__(
            self, 
            csv_path:str, 
            image_dir:str, 
            label_dir:str, 
            classes:List[str], 
            preprocessing_transforms = None, 
            augmentation_transforms = None, 
            train=True,
            patch_based = False
            ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.classes = classes
        self.preprocessing_transforms = preprocessing_transforms
        self.augmentation_transforms = augmentation_transforms
        self.train = train
        self.patch_based = patch_based
        
        self.images = []

        with open(csv_path, "r", newline='') as f:
            self.reader = csv.reader(f, delimiter="#")
            for row in self.reader:
                self.images.append(os.path.join(row[0]))
        
        if self.patch_based:
            self.subjects_dataset = None
            
            subjects = []
            with open(csv_path, "r", newline='') as f:
                self.reader = csv.reader(f, delimiter="#")
                for row in self.reader:

                    filename = row[0][:-4]

                    mask_names = [filename for filename in os.listdir(os.path.join(label_dir, filename))]

                    dct = {"Channel"+str(i) : tio.LabelMap(filepath) for (i, filepath) in zip(range(len(mask_names)), [os.path.join(label_dir, filename, fname) for fname in mask_names])}
                    dct["image"] = tio.ScalarImage(os.path.join(image_dir, row[0]))
                    
                    subjects.append(tio.Subject(dct))
            
            if self.train:
                transform = tio.Compose([
                    self.preprocessing_transforms,
                    self.augmentation_transforms
                ])
            else:
                transform = self.preprocessing_transforms

            self.subjects_dataset = tio.SubjectsDataset(
                subjects = subjects,
                transform = transform
            )

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = tio.ScalarImage(os.path.join(self.image_dir, self.images[idx]))
        subject = tio.Subject(image = image)
        for label in os.listdir(os.path.join(self.label_dir, os.path.splitext(self.images[idx])[0])):
            subject.add_image(image = tio.LabelMap(os.path.join(self.label_dir, os.path.splitext(self.images[idx])[0], label)), 
                            image_name = str(os.path.splitext(os.path.splitext(label)[0])[0]))
        subject = self.preprocessing_transforms(subject)
        # if self.train:
        #     subject = self.augmentation_transforms(subject)
        # print(self.images[idx])
        return torch.as_tensor(subject['image'][tio.DATA], dtype=torch.float32), torch.cat([subject[key][tio.DATA] for key in list(subject.keys())[1:]])
    
    def get_subjects_dataset(self):
        return self.subjects_dataset


class CanineLesionsUniformSampler(tio.data.UniformSampler):
    
    def _generate_patches(
        self,
        subject: Subject,
        num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        valid_range = subject.spatial_shape - self.patch_size
        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            i, j, k = tuple(int(torch.randint(x + 1, (1,)).item()) for x in valid_range)
            index_ini = i, j, k
            labels = []
            for k in subject.keys():
                if k == "image":
                    img = self.extract_patch(tio.Subject(image=subject[k]), index_ini)['image'].data
                else:
                    label = self.extract_patch(tio.Subject(image=subject[k]), index_ini)['image'].data
                    labels.append(label)
            labels = torch.cat(labels, dim=0)
            yield img, labels
            if num_patches is not None:
                patches_left -= 1
