import os
import re
import SimpleITK as sitk
import pandas as pd
import csv
import sys
import copy

from Python.utils.utils import *

import logging
logging.basicConfig(level=logging.INFO)


TEST_SIZE = 5


def get_classes_overview(nrrd_source_folderpath):
    """
    Returns an overview of classes in the entire, UNFILTERED dataset.
    """
    classes = {}

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    for filename in os.listdir(nrrd_source_folderpath):
        filepath = os.path.join(nrrd_source_folderpath, filename)
        # filepath = source_folderpath+'/'+filename
        reader.SetFileName(filepath)
        image = reader.Execute()
        # image = sitk.ReadImage(filepath)

        for key in image.GetMetaDataKeys():
            if re.match(r'^Segment\d_Name$', key):                
                if image.GetMetaData(key).capitalize() not in list(classes.keys()):
                    classes[image.GetMetaData(key).capitalize()] = 1
                else:
                    classes[image.GetMetaData(key).capitalize()] += 1

    df = pd.DataFrame.from_dict(classes, orient='index', columns=['Total occurences']).sort_values('Total occurences', ascending=False)
    return df


def compare_seg_and_mha_sizes(image_sizes, segmentation_sizes):
    logger = logging.getLogger(name="compare_seg_and_mha_sizes")
    logger.info(" Checking segmentation-image pairs' sizes...")

    """
    Compares sizes of segmentation and MHA file. If the sizes do not match, the segmentation is incomplete, or the file is corrupted.
    """
    matching_indices = []
    sizes_matching = 0
    sizes_not_matching = 0
    for i in range(len(image_sizes)):
        if image_sizes[i] == segmentation_sizes[i]:
            sizes_matching += 1
            matching_indices.append(i)
        else:
            sizes_not_matching += 1
    logger.info(f" Samples with matching sizes: {sizes_matching}")
    logger.info(f" Samples with different sizes: {sizes_not_matching}")
    logger.info(f" Size checking completed.")
    return sizes_matching, sizes_not_matching, matching_indices


def count_correct_incorrect_cases(source_folderpath):
    logger = logging.getLogger(name="count_correct_incorrect_cases")
    logger.info(f" Counting data...")
    """
    Loops over all files in the folder to check how many of them can be read and loaded, and counts occurences 
    """

    reader = sitk.ImageFileReader()

    nrrd_files_read = 0
    mha_files_read = 0

    incorrect_cases_no_segment_x = 0
    incorrect_cases_bad_segmentation_size = 0
    incorrect_cases_nrrd_file_not_read = 0
    incorrect_cases_mha_file_not_read = 0


    reader.SetImageIO("NrrdImageIO")
    segmentation_sizes = []

    dir_len = len(os.listdir(os.path.join(source_folderpath, "correct_seg_nrrd")))

    for i, filename in enumerate(sorted(os.listdir(os.path.join(source_folderpath, "correct_seg_nrrd")))):
        sys.stdout.write(f"\rReading segmentation {i}/{dir_len}")
        sys.stdout.flush()
        filepath = os.path.join(source_folderpath, "correct_seg_nrrd", filename)
        try:
            reader.SetFileName(filepath)
            image = reader.Execute()
            segmentation_sizes.append(str(image.GetSize()))
            case_incorrect_no_segment_x = True

            for key in image.GetMetaDataKeys():
                if re.match(r'^Segment\d_Name$', key):
                    case_incorrect_no_segment_x = False
            
            nrrd_files_read += 1
            if case_incorrect_no_segment_x:
                incorrect_cases_no_segment_x += 1
        except:
            print()
            logger.warning(f" Failed while reading file {filename}")
            incorrect_cases_nrrd_file_not_read += 1
    logger.info(f" Segmentations read.") 
    
    reader.SetImageIO("MetaImageIO")
    image_sizes = []
    for i, filename in enumerate(sorted(os.listdir(os.path.join(source_folderpath, "correct_mha")))):
        sys.stdout.write(f"\rReading images {i}/{dir_len}")
        sys.stdout.flush()
        filepath = os.path.join(source_folderpath, "correct_mha", filename)
        try:
            reader.SetFileName(filepath)
            image = reader.Execute()
            image_sizes.append(str(image.GetSize()))
            mha_files_read += 1
        except:
            incorrect_cases_mha_file_not_read += 1
            print()
            logger.warning(f" Failed while reading file {filename}")
            image_sizes.append("Error")
    
    logger.info(f" Images read.")

    sizes_matching, sizes_not_matching, correct_indices = compare_seg_and_mha_sizes(image_sizes, segmentation_sizes)

    return correct_indices


def calculate_classes_volume(source_folderpath, testing=False):
    logger = logging.getLogger("calculate_classes_volume")
    logger.info(f" Calculating diseases' volumes...")
    """
    Calculate volume of each class - pixelwise and volumewise.
    """
    mha_folder = "correct_mha_filtered"
    seg_folder = "correct_seg_nrrd_filtered"
    classes_volumes_pixelwise = get_empty_classes_dict(os.path.join(source_folderpath, seg_folder))
    classes_volumes_volumewise = copy.deepcopy(classes_volumes_pixelwise)
    classes_occurences = copy.deepcopy(classes_volumes_pixelwise)
    classes_biggest_occurences_pixels = copy.deepcopy(classes_volumes_pixelwise)
    classes_smallest_occurences_pixels = copy.deepcopy(classes_volumes_pixelwise)
    classes_biggest_occurences_cubic_mm = copy.deepcopy(classes_volumes_pixelwise)
    classes_smallest_occurences_cubic_mm = copy.deepcopy(classes_volumes_pixelwise)
    classes_all_occurences_pixels = {}
    classes_all_occurences_cubic_mm = {}
    for key in classes_biggest_occurences_pixels.keys():
        classes_biggest_occurences_pixels[key] = 0
        classes_smallest_occurences_pixels[key] = 100000000
        classes_biggest_occurences_cubic_mm[key] = np.NaN
        classes_smallest_occurences_cubic_mm[key] = np.NaN
        classes_all_occurences_pixels[key] = []
        classes_all_occurences_cubic_mm[key] = []
    seg_not_matching = []

    # images = sorted(os.listdir(os.path.join(source_folderpath, mha_folder)))
    segmentations = sorted(os.listdir(os.path.join(source_folderpath, seg_folder)))
    with open("Results/individual_disease_volumes.csv", "w") as f:
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        writer = csv.writer(f, delimiter=",")
        if testing:
            rg = TEST_SIZE
        else:
            rg = len(segmentations)
        for i in range(rg):
            sys.stdout.write(f"\rCalculating for file {i+1} of {len(segmentations)}...")
            sys.stdout.flush()
        # for i in range(len(segmentations)):
            # image = sitk.ReadImage(os.path.join(source_folderpath, mha_folder, images[i]))
            reader.SetFileName(os.path.join(source_folderpath, seg_folder, segmentations[i]))
            seg = reader.Execute()
            labels_classes = extract_diseases_in_image(seg)
            spacing = seg.GetSpacing()

            voxel_volume = spacing[0]*spacing[1]*spacing[2]

            seg_arr = sitk.GetArrayFromImage(seg)

            row = []

            uniques, counts = np.unique(seg_arr, return_counts=True)
            
            if not len(uniques) - 1 == len(labels_classes):
                seg_not_matching.append(segmentations[i])
            
            
            row.append(os.path.splitext(os.path.splitext(segmentations[i])[0])[0])

            for key in labels_classes.keys():
                classes_volumes_pixelwise[labels_classes[key]] += counts[key]
                classes_volumes_volumewise[labels_classes[key]] += counts[key]*voxel_volume
                classes_occurences[labels_classes[key]] += 1

            for i in range(1, len(labels_classes.keys()) + 1):
                classes_all_occurences_pixels[labels_classes[i]].append(counts[i])
                classes_all_occurences_cubic_mm[labels_classes[i]].append(counts[i]*voxel_volume)
                row.append(labels_classes[i].capitalize())
                row.append(counts[i])
                row.append(int(counts[i]*voxel_volume))

        for key in classes_all_occurences_cubic_mm.keys():
            classes_biggest_occurences_cubic_mm[key] = max(classes_all_occurences_cubic_mm[key])
            classes_smallest_occurences_cubic_mm[key] = min(classes_all_occurences_cubic_mm[key])
            classes_biggest_occurences_pixels[key] = max(classes_all_occurences_pixels[key])
            classes_smallest_occurences_pixels[key] = min(classes_all_occurences_pixels[key])
        writer.writerow(row)
        
        print()
        f.close()

    with open("Results/all_occurences.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        for key in classes_all_occurences_pixels.keys():
            row = []
            row.append(key.capitalize())
            row.append("Biggest occurence in pixels")
            row.append(classes_biggest_occurences_pixels[key])
            row.append("Biggest occurence in cubic milimeters")
            row.append(classes_biggest_occurences_cubic_mm[key])
            row.append("Smallest occurence in pixels")
            row.append(classes_smallest_occurences_pixels[key])
            row.append("Smallest occurence in cubic milimeters")
            row.append(classes_smallest_occurences_cubic_mm[key])
            row.append("Std dev in pixels")
            row.append(np.std(np.array(classes_all_occurences_pixels[key])))
            row.append("Std dev in cubic milimeters")
            row.append(np.std(np.array(classes_all_occurences_cubic_mm[key])))
            writer.writerow(row)
            row = classes_all_occurences_pixels[key]
            writer.writerow(row)
            row = classes_all_occurences_cubic_mm[key]
            writer.writerow(row)
        f.close()

    with open("Results/total_disease_volumes.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        row = ["Disease", "Pixels", "Cubic milimeters"]
        writer.writerow(row)
        for key in classes_volumes_pixelwise.keys():
            # print(f"{key} : {classes_volumes_pixelwise[key]} pixels, {classes_volumes_volumewise[key]} cubic milimeters")
            row = [key.capitalize(), classes_volumes_pixelwise[key], classes_volumes_volumewise[key]]
            writer.writerow(row)
        f.close()

    logger.info(f" Statistics saved to total_disease_volumes.csv")
    with open("Results/disease_occurences.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        row = ["Disease", "Occurences"]
        writer.writerow(row)
        for key in classes_occurences.keys():
            row = [key.capitalize(), classes_occurences[key]]
            writer.writerow(row)
        f.close()

    logger.info(f" Occurrences counts saved to disease_occurences.csv")
    return 0

