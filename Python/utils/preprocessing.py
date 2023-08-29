import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")
import SimpleITK as sitk
import os
import numpy as np
import shutil
import logging
import json
import time
from Python.utils.utils import extract_diseases_in_image
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)

TEST_SIZE = 5

def load_data(source_folderpath, testing=False):

    logger = logging.getLogger("load_data")
    logger.addHandler(ch)
    logger.propagate = False

    logger.info(" ==============================")
    logger.info(f" Loading data...")
    logger.info(" ==============================")

    mha_folder = "correct_mha_filtered"
    seg_folder = "correct_seg_nrrd_filtered"

    images = sorted(os.listdir(os.path.join(source_folderpath, mha_folder)))
    segmentations = sorted(os.listdir(os.path.join(source_folderpath, seg_folder)))

    if testing:
        images = images[:TEST_SIZE]
        segmentations = segmentations[:TEST_SIZE]

    images_loaded = []
    segmentations_loaded = []

    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")

    for i, image in enumerate(images):
        sys.stdout.write(f"\rLoading image {i+1} of {len(images)}...")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folderpath, mha_folder, image))
        images_loaded.append(reader.Execute())
    print()
    logger.info(f" {len(images_loaded)} images loaded.")
    logger.info(" ==============================")
    logger.info(f" Loading segmentations...")
    logger.info(" ==============================")

    reader.SetImageIO("NrrdImageIO")
    for i, seg in enumerate(segmentations):
        sys.stdout.write(f"\rLoading segmentation {i+1} of {len(segmentations)}...")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folderpath, seg_folder, seg))
        segmentations_loaded.append(reader.Execute())
    print()
    logger.info(f" {len(segmentations_loaded)} segmentations loaded.")
    return images_loaded, segmentations_loaded, images, segmentations


def check_metadata_consistency(source_folderpath):

    consistent_origin, inverted_origin, inconsistent_origin = check_origin_consistency(source_folderpath, load_data(source_folderpath))

    consistent_spacing, inconsistent_spacing = check_spacing_consistency(source_folderpath)

    consistent_direction, inconsistent_direction = check_direction_consistency(source_folderpath)

    print(f"========== DATA STATISTICS ==========")
    print(f"Samples with consistent origin: {consistent_origin}")
    print(f"Samples with inverted origin: {inverted_origin}")
    print(f"Samples with inconsistent origin: {inconsistent_origin}")    
    print(f"Samples with consistent spacing: {consistent_spacing}")
    print(f"Samples with inconsistent spacing: {inconsistent_spacing}")
    print(f"Samples with consistent direction: {consistent_direction}")
    print(f"Samples with inconsistent direction: {inconsistent_direction}")
    print("=======================================")


def check_origin_consistency(source_folderpath, loaded_data):
    logger = logging.getLogger("check_origin_consistency")
    logger.addHandler(ch)
    logger.propagate = False

    images_loaded, segmentations_loaded, images, segmentations = loaded_data

    consistent_origin = 0
    inverted_origin = 0
    inconsistent_origin = 0

    logger.info(" ==============================")
    logger.info(f" Performing origin consistency check...")
    logger.info(" ==============================")

    img_writer = sitk.ImageFileWriter()

    for i in range(len(images_loaded)):
        sys.stdout.write(f"\rChecking image {i+1} of {len(images_loaded)}...")
        sys.stdout.flush()
        if images_loaded[i].GetOrigin() == segmentations_loaded[i].GetOrigin():
            shutil.copy(os.path.join(source_folderpath, "correct_mha_filtered", images[i]), 
                        os.path.join(source_folderpath, "final-data", "origin-checked", "images"))
            
            shutil.copy(os.path.join(source_folderpath, "correct_seg_nrrd_filtered", segmentations[i]), 
                        os.path.join(source_folderpath, "final-data", "origin-checked", "segmentations"))
            consistent_origin += 1

        elif max(abs(abs(np.array(images_loaded[i].GetOrigin())) - abs(np.array(segmentations_loaded[i].GetOrigin())))) < 1e-5:
            image_arr = sitk.GetArrayFromImage(images_loaded[i])
            seg_arr = sitk.GetArrayFromImage(segmentations_loaded[i])

            image = sitk.GetImageFromArray(image_arr)
            image.SetOrigin(segmentations_loaded[i].GetOrigin())
            image.SetDirection(segmentations_loaded[i].GetDirection())
            image.SetSpacing(segmentations_loaded[i].GetSpacing())

            seg = sitk.GetImageFromArray(seg_arr)
            seg.SetOrigin(segmentations_loaded[i].GetOrigin())
            seg.SetDirection(segmentations_loaded[i].GetDirection())
            seg.SetSpacing(segmentations_loaded[i].GetSpacing())

            img_writer.SetImageIO("MetaImageIO")
            img_writer.SetFileName(fileName=os.path.join(source_folderpath, "final-data", "origin-checked", "images", images[i]))
            img_writer.Execute(image)

            img_writer.SetImageIO("NrrdImageIO")
            img_writer.SetFileName(fileName=os.path.join(source_folderpath, "final-data", "origin-checked", "segmentations", segmentations[i]))
            img_writer.UseCompressionOn()
            img_writer.Execute(seg)

            inverted_origin += 1
        else:   # the image needs resampling
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(segmentations_loaded[i])
            resampler.SetInterpolator(sitk.sitkBSpline)
            image = resampler.Execute(images_loaded[i])

            img_writer.SetImageIO("MetaImageIO")
            img_writer.SetFileName(fileName=os.path.join(source_folderpath, "final-data", "origin-checked", "images", images[i]))
            img_writer.Execute(image)

            shutil.copy(os.path.join(source_folderpath, "correct_seg_nrrd_filtered", segmentations[i]), 
                        os.path.join(source_folderpath, "final-data", "origin-checked", "segmentations"))

            inconsistent_origin += 1
    print()
    logger.info(f" Origin consistency check finished.")

    return consistent_origin, inverted_origin, inconsistent_origin


def check_spacing_consistency(source_folderpath, testing=False):

    logger = logging.getLogger("check_spacing_consistency")
    logger.addHandler(ch)
    logger.propagate = False

    consistent_spacing = 0
    inconsistent_spacing = 0

    logger.info(" ==============================")
    logger.info(" Performing spacing consistency check...")
    logger.info(" ==============================")

    images = sorted(os.listdir(os.path.join(source_folderpath, "final-data", "origin-checked", "images")))
    segmentations = sorted(os.listdir(os.path.join(source_folderpath, "final-data", "origin-checked", "segmentations")))

    if testing:
        images = images[:TEST_SIZE]
        segmentations = segmentations[:TEST_SIZE]
    
    images_loaded = []
    segmentations_loaded = []
    
    reader = sitk.ImageFileReader()

    reader.SetImageIO("MetaImageIO")
    for i, image in enumerate(images):
        sys.stdout.write(f"\rLoading image {i+1} of {len(images)}...")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folderpath, "final-data", "origin-checked", "images", image))
        images_loaded.append(reader.Execute())
    print()
    logger.info(f" {len(images)} images loaded.")

    reader.SetImageIO("NrrdImageIO")
    for i, seg in enumerate(segmentations):
        sys.stdout.write(f"\rLoading segmentation {i+1} of {len(segmentations)}...")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folderpath, "final-data", "origin-checked", "segmentations", seg))
        segmentations_loaded.append(reader.Execute())
    print()
    logger.info(f" {len(segmentations)} segmentations loaded.")

    img_writer = sitk.ImageFileWriter()
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    for i in range(len(images)):
        sys.stdout.write(f"\rChecking image {i+1} of {len(images)}...")
        sys.stdout.flush()
        if np.sum(np.array(images_loaded[i].GetSpacing())) - np.sum(np.array(segmentations_loaded[i].GetSpacing())) < 1e-5:
            shutil.copy(os.path.join(source_folderpath, "final-data", "origin-checked", "images", images[i]),
                        os.path.join(source_folderpath, "final-data", "spacing-checked", "images"))
            
            shutil.copy(os.path.join(source_folderpath, "final-data", "origin-checked", "segmentations", segmentations[i]),
                        os.path.join(source_folderpath, "final-data", "spacing-checked", "segmentations"))
            consistent_spacing += 1
        else:
            resampler.SetReferenceImage(segmentations_loaded[i])
            image = resampler.Execute(images_loaded[i])

            img_writer.SetImageIO("MetaImageIO")
            img_writer.SetFileName(os.path.join(source_folderpath, "final-data", "spacing-checked", "images", images[i]))
            img_writer.Execute(image)

            shutil.copy(os.path.join(source_folderpath, "final-data", "origin-checked", "segmentations", segmentations[i]),
                        os.path.join(source_folderpath, "final-data", "spacing-checked", "segmentations"))
            inconsistent_spacing += 1
    print()
    logger.info(" Spacing consistency check finished.")

    return consistent_spacing, inconsistent_spacing


def check_direction_consistency(source_folderpath, testing=False):
    
    logger = logging.getLogger("check_direction_consistency")
    logger.addHandler(ch)
    logger.propagate = False

    logger.info(" ==============================")
    logger.info(f" Performing direction consistency check...")
    logger.info(" ==============================")
    
    images = sorted(os.listdir(os.path.join(source_folderpath, "final-data", "spacing-checked", "images")))
    segmentations = sorted(os.listdir(os.path.join(source_folderpath, "final-data", "spacing-checked", "segmentations")))

    images_loaded = []
    segmentations_loaded = []

    if testing:
        images = images[:TEST_SIZE]
        segmentations = segmentations[:TEST_SIZE]

    consistent_direction = 0
    inconsistent_direction = 0

    reader = sitk.ImageFileReader()
    
    reader.SetImageIO("MetaImageIO")
    for i, image in enumerate(images):
        sys.stdout.write(f"\rLoading image {i+1} of {len(images)}...")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folderpath, "final-data", "spacing-checked", "images", image))
        images_loaded.append(reader.Execute())
    print()
    logger.info(f" {len(images)} images loaded.")
    reader.SetImageIO("NrrdImageIO")
    for i, seg in enumerate(segmentations):
        sys.stdout.write(f"\rLoading segmentation {i+1} of {len(segmentations)}...")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folderpath, "final-data", "spacing-checked", "segmentations", seg))
        segmentations_loaded.append(reader.Execute())
    print()
    logger.info(f" {len(segmentations)} segmentations loaded.")

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)

    img_writer = sitk.ImageFileWriter()
    img_writer.SetImageIO("MetaImageIO")

    for i in range(len(images)):
        sys.stdout.write(f"\rChecking image {i+1} of {len(images)}...")
        sys.stdout.flush()
        if np.all(np.array(list(images_loaded[i].GetDirection())) == np.array(list(segmentations_loaded[i].GetDirection()))):
            shutil.copy(os.path.join(source_folderpath, "final-data", "spacing-checked", "images", images[i]),
                        os.path.join(source_folderpath, "final-data", "all-checked", "images"))
            shutil.copy(os.path.join(source_folderpath, "final-data", "spacing-checked", "segmentations", segmentations[i]),
                        os.path.join(source_folderpath, "final-data", "all-checked", "segmentations"))
            consistent_direction += 1
        else:
            resampler.SetReferenceImage(segmentations_loaded[i])
            image = resampler.Execute(images_loaded[i])

            img_writer.SetFileName(os.path.join(source_folderpath, "final-data", "all-checked", "images", images[i]))
            img_writer.Execute(image)

            shutil.copy(os.path.join(source_folderpath, "final-data", "spacing-checked", "segmentations", segmentations[i]),
                        os.path.join(source_folderpath, "final-data", "all-checked", "segmentations"))
            inconsistent_direction += 1
    print()
    logger.info(f" Direction consistency check completed.")
    logger.info(f" Preprocessed data saved.")
    return consistent_direction, inconsistent_direction


def unify_spacings(source_folder_path:str, spacing:List[float] = [1.0, 1.0, 1.0]):
    logger = logging.getLogger("unify_spacings")
    logger.addHandler(ch)
    logger.propagate = False

    logging.info(f" Unifying spacings in the images and segmentations - chosen spacings: {spacing}")
    # filenames = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(os.path.join(source_folder_path, "final-data", "all-checked", "images")))]
    filenames = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(os.path.join(source_folder_path, "correct_mha")))]

    l = len(filenames)
    
    img_reader = sitk.ImageFileReader()
    
    # resampler.SetInterpolator(sitk.sitkBSpline)
    # resampler.SetOutputSpacing(spacing)

    img_writer = sitk.ImageFileWriter()
    img_writer.UseCompressionOn()
    for i, filename in enumerate(filenames):
        sys.stdout.write(f"\rProcessing image {i+1} / {l}")
        sys.stdout.flush()

        img_reader.SetImageIO("MetaImageIO")
        img_reader.SetFileName(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "images", filename+".mha"))

        image = img_reader.Execute()

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, spacing)]
        image = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkBSpline, image.GetOrigin(), spacing, image.GetDirection(), 0, image.GetPixelID())
        img_writer.SetFileName(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "unit-spacing-images", filename+".mha"))
        img_writer.SetImageIO("MetaImageIO")
        img_writer.Execute(image)

        img_reader.SetImageIO("NrrdImageIO")
        img_reader.SetFileName(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "segmentations", filename+".seg.nrrd"))

        image = img_reader.Execute()

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        image = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkBSpline, image.GetOrigin(), spacing, image.GetDirection(), 0, image.GetPixelID())
        img_writer.SetFileName(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "unit-spacing-segmentations", filename+".seg.nrrd"))
        img_writer.SetImageIO("NrrdImageIO")
        img_writer.Execute(image)

    print()
    logging.info(f" Spacings unified.")
    return


def prepare_segmentation_dirs(source_folderpath):
    logger = logging.getLogger("prepare_segmentation_dirs")
    logger.addHandler(ch)
    logger.propagate = False

    logger.info(" Preparing directories for segmentation masks...")
    l = len(os.listdir(os.path.join(source_folderpath, "final-data", "all-checked", "images")))
    for i, file in enumerate(os.listdir(os.path.join(source_folderpath, "final-data", "all-checked", "images"))):
        sys.stdout.write(f"\r{i+1} / {l}")
        sys.stdout.flush()
        filename = os.path.splitext(file)[0]
        if not os.path.exists(os.path.join(source_folderpath, "final-data", "all-checked", "masks", filename)):
            os.mkdir(os.path.join(source_folderpath, "final-data", "all-checked", "masks", filename))
    print()
    logger.info(f" Directories ready.")


def prepare_segmentation_masks(source_folderpath, classes_to_channels, custom_classes):
    logger = logging.getLogger("prepare_segmentation_masks")
    logger.addHandler(ch)
    logger.propagate = False
    
    logger.info(f" Preparing segmentation masks...")
    filenames = sorted([os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(source_folderpath, "final-data", "all-checked", "images"))])

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")

    img_writer = sitk.ImageFileWriter()
    img_writer.SetImageIO("NrrdImageIO")
    img_writer.UseCompressionOn()
    l = len(filenames)
    for i, filename in enumerate(filenames):
        t1 = time.time()
        reader.SetFileName(os.path.join(source_folderpath, "final-data", "all-checked", "segmentations", filename+".seg.nrrd"))
        seg_original = reader.Execute()
        seg_arr = sitk.GetArrayFromImage(seg_original)
        jsonfile = open(os.path.join(source_folderpath, "final-data", "all-checked", "seg-json", filename+".json"), "r")
        seg_json = json.load(jsonfile)

        for k, v in classes_to_channels.items():    # iterate over channels
            arr = np.zeros(seg_arr.shape)
            for k1, v1 in seg_json['lesions'].items():  # iterate over lesions 
                if k == k1:
                    arr = np.array((seg_arr==v1[0]).astype(int))
            img_writer.SetFileName(os.path.join(source_folderpath, "final-data", "all-checked", "masks", filename, "Channel"+str(v)+".seg.nrrd"))
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(seg_original.GetSpacing())
            img.SetOrigin(seg_original.GetOrigin())
            img.SetDirection(seg_original.GetDirection())
            img_writer.Execute(img)
        
        next_channel_index = len(classes_to_channels)

        for custom_class in custom_classes:
            arr = np.zeros(seg_arr.shape)
            for k1, v1 in seg_json['lesions'].items():
                if custom_class == k1:
                    arr = np.array((seg_arr==v1[0]).astype(int))
            img_writer.SetFileName(os.path.join(source_folderpath, "final-data", "all-checked", "masks", filename, "Channel"+str(next_channel_index)+".seg.nrrd"))
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(seg_original.GetSpacing())
            img.SetOrigin(seg_original.GetOrigin())
            img.SetDirection(seg_original.GetDirection())
            img_writer.Execute(img)
            next_channel_index += 1
        # for j in range(total_classes_count):    # iterate over all channels in image - all segmentation masks
        #     arr = np.zeros(seg_arr.shape)
        #     for k, v in seg_json['lesions'].items():    # iterate over all lesions in the current image
        #         if v[1] == j:
        #             arr = np.array((seg_arr==v[0]).astype(int))
        #     img_writer.SetFileName()
        #     img_writer.Execute(sitk.GetImageFromArray(arr))
        t2 = time.time()
        sys.stdout.write(f"\r{i+1} / {l}, previous sample time elapsed: {t2-t1} seconds")
        sys.stdout.flush()
    print()
    logger.info(f" Segmentation masks ready.")
    # below - for testing purposes
    # print()
    # print()

    # reader.SetFileName(os.path.join(source_folderpath, "final-data", "all-checked", "masks", filename, "Channel"+str(10)))
    # arrtest = reader.Execute()
    # print(f"Uniques and counts in seg_arr: {np.unique(seg_arr, return_counts=True)}")
    # print(f"Uniques and counts for lung mass(1): {np.unique(sitk.GetArrayFromImage(arrtest), return_counts=True)}")
    return 0