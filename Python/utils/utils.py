import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")
import os
import re
import shutil
import logging
import json
import time
import numpy as np
import SimpleITK as sitk

from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)

def extract_correct_data(source_folder_path:str, correct_indices:list) -> None:
    """
    Obsolete, not used anymore.

    Separates segmentation and image files from the given location, and copies them to their corresponding folders.
    
    Parameters
    ----------
    source_folder_path : str
        Path to folder contatining not-separated files.
    correct_indices : list
        List of indices to correct files inside the folder.
    
    """
    logger = logging.getLogger("extract_correct_data")
    logger.addHandler(ch)
    logger.propagate = False
    
    indices_len = len(correct_indices)
    nrrd_images = sorted(os.listdir(os.path.join(source_folder_path, "correct_seg_nrrd")))
    mha_images = sorted(os.listdir(os.path.join(source_folder_path, "correct_mha")))
    logger.info(f" Extracting correct data...")
    for i, index in enumerate(correct_indices):
        sys.stdout.write(f"\rCopying NRRD image {i}/{indices_len}")
        shutil.copy(os.path.join(source_folder_path, "correct_seg_nrrd", nrrd_images[index]), 
                    os.path.join(source_folder_path, "correct_seg_nrrd_filtered"))
        sys.stdout.write(f"\nCopying MHA image {i}/{indices_len}")
        sys.stdout.flush()
        shutil.copy(os.path.join(source_folder_path, "correct_mha", mha_images[index]), 
                    os.path.join(source_folder_path, "correct_mha_filtered"))
    print()
    logger.info(f" Copying finished.")


def get_empty_classes_dict(destination_folderpath:str, testing:bool=False) -> Dict[str, int]:
    """
    Returns a dictionary of all existing classes (lesions) in the dataset,
    paired with their volumes set to 0.

    Parameters
    ---------
    destination_folderpath : str
        Folder contatining segmentation files.
    testing : bool
        Flag indication whether the program is running in test mode, subsetting the data.
        Mainly used for development.
    
    Returns
    ------
    A dictionary of all the classes paired with zeros. Useful representation for further processing.
    """

    logger = logging.getLogger("get_empty_classes_dict")
    logger.addHandler(ch)
    logger.propagate = False
    
    logger.info(f" Extracting lesions in the dataset...")
    classes_volumes = {}
    
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")

    for i, filename in enumerate(os.listdir(destination_folderpath)):
        sys.stdout.write(f"\rChecking segmentation file {i+1} of {len(os.listdir(destination_folderpath))}")
        sys.stdout.flush()
        filepath = destination_folderpath+'/'+filename
        reader.SetFileName(filepath)
        image = reader.Execute()
    
        for k in image.GetMetaDataKeys():
            if re.match(r'^Segment\d_Name$', k):
                classes_volumes[image.GetMetaData(k)] = 0
    print()
    logger.info(f" {len(classes_volumes)} different diseases discovered.")
    return classes_volumes


def extract_segmentation_slices(seg:np.ndarray) -> np.ndarray:
    """
    Returns indices of slices containing any segmentation. Used for visualization.

    Parameters
    ---------
    seg : np.ndarray
        Segmentation matrix, extracted from segmentation file.

    Returns
    -------
    Array of indices to slices containing any segmenatation.
    """
    
    indices = []
    for i in range(seg.shape[0]):
        if len(np.unique(seg[i])) > 1:
            indices.append(i)
    return np.array(indices)


def extract_diseases_in_image(seg:sitk.Image) -> Dict[int, str]:
    """
    Assigns integer labels to lesions occuring in the image.\n
    The labels are assigned following order of lesions' occurence in the segmentation.\n
    Example:\n
    ...\n
    Segment0_ID:=Segment_1\n
    Segment0_LabelValue:=1\n
    Segment0_Layer:=0\n
    Segment0_Name:=lipoma\n
    ...\n
    Segment1_ID:=Segment_2\n
    Segment1_LabelValue:=2\n
    Segment1_Layer:=0\n
    Segment1_Name:=subcutaneous mass\n
    ...\n

    Returned dictionary would look like this:\n
    {
        1 : lipoma,
        2 : subcutaneous mass
    }

    Parameters
    --------
    seg : sitk.Image
        SimpleITK Image file, being a segmentation file in this context.

    Returns
    -------
    Dictionary of integer labels paired with lesions found in the image.
    """
    i = 1

    labels_diseases = {}

    for k in seg.GetMetaDataKeys():
        if re.match(r'^Segment\d_Name$', k):
            labels_diseases[i] = seg.GetMetaData(k)
            i += 1

    return labels_diseases


def count_samples_in_classes(source_folder_path:str) -> Dict[str, int]:
    """
    Counts the number of occurences of each lesion in the dataset.

    Parameters
    ------
    source_folder_path : str
        Path to the root of the data folder.
    
    Returns
    ------
    Dictionary of lesion and number of its occurences pairs.
    """
    logger = logging.getLogger("count_samples_in_classes")
    logger.addHandler(ch)
    logger.propagate = False

    dct = get_empty_classes_dict(os.path.join(source_folder_path, "correct_seg_nrrd"))

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")

    logger.info(f" Counting samples in classes...")
    for i, filename in enumerate(os.listdir(os.path.join(source_folder_path, "correct_seg_nrrd"))):
        sys.stdout.write(f"\r{i+1} / {len(os.listdir(os.path.join(source_folder_path, 'correct_seg_nrrd')))}")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folder_path, "correct_seg_nrrd", filename))
        image = reader.Execute()
        for k in image.GetMetaDataKeys():
            if re.match(r'^Segment\d_Name$', k):
                dct[image.GetMetaData(k).lower()] += 1
    print()
    logger.info(f" {len(os.listdir(os.path.join(source_folder_path, 'correct_seg_nrrd')))} files scanned.")
    
    return dct


def encode_lesions(source_folder_path:str) -> Dict[str, int]:
    """
    Assigns unique integer labels to lesions.

    Parameters
    --------
    source_folder_path : str
        Path to the root of the data folder.

    Returns
    -------
    dct : Dict[str, int]
        Dictionary of lesions paired with integer labels.
    """
    dct = {}
    for i, key in enumerate(get_empty_classes_dict(os.path.join(source_folder_path, "correct_seg_nrrd")).keys()):
        dct[key] = i
    
    return dct


def merge_classes(encoded_lesions:Dict[str, int], classes:list, custom_classes:Optional[list]=None, keywords:Optional[list]=None) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, int]]:
    """
    Carries out a class merge. 
    
    Merging classes is useful in multi-label, multi-class segmentation, for more general segmentation.
    Each of custom_classes is a class added to the dataset, composed from different classes already present.
    The classes are merged depending on whether the passed keyword matches the name of a certain class.
    For example, passing a keyword "node" would match classes like "sternal lymph node", "axillary lymph node" etc.

    Parameters
    --------
    encoded_lesions : str
        Dictionary of lesions with assigned integer labels, generated by the encode_lesions() function.
    classes : list
        List of already present classes to be chosen from the dataset.
    custom_classes : list
        List of custom classes to be created.
    keywords : list
        List of keywords used to merge classes.
    
    Returns
    -------
    classes_to_labels : Dict[str, int]
        Dictionary of classes paired with their respective integer encodings. Contains both original and custom classes.
    keywords_to_custom_classes : Dict[str, str]
        Dictionary of keywords to custom classes paired with those custom classes.
    classes_to_channels : Dict[str, int]
        Dictionary of classes paired with their respective channels in segmentation.
    """
    logger = logging.getLogger("merge_classes")
    logger.addHandler(ch)
    logger.propagate = False

    if custom_classes:
        logger.info(f" Beginning class merge. Found {len(custom_classes)} custom classes.")
    classes_to_labels = {}
    
    for key in encoded_lesions.keys():
        if key in classes:
            classes_to_labels[key] = encoded_lesions[key]

    if not custom_classes and keywords:
        raise TypeError(" No custom classes were provided, but keywords regarding merging are present. Please remove unnecessary keywords, or provide custom classes.")
    elif custom_classes and not keywords:
        raise TypeError(" Custom classes are present, but no keywords regarding merging were provided. Please provide appropiate keywords.")
    elif custom_classes and keywords and len(custom_classes) == len(keywords):
        for j, custom_class in enumerate(custom_classes):
            classes_to_labels[custom_class] = max(encoded_lesions.values()) + 1
    
    if custom_classes:
        logger.info(f" Class merge completed.")
        keywords_to_custom_classes = {}
        for i in range(len(custom_classes)):
            keywords_to_custom_classes[keywords[i]] = custom_classes[i]
    
    # for key in sample_counts.keys():
    #     if sample_counts[key] >= min_samples:
    #         classes_to_labels[key] = encoded_lesions[key]
    #     elif key in classes:
    #         classes_to_labels[key] = encoded_lesions[key]
    
    
            # accu = 0
            # for key in sample_counts.keys():
            #     if re.findall(r"\b" + keywords[j] + r"\b", key):
            #         accu += sample_counts[key]
            # if accu >= min_samples:
    classes_to_channels = {}
    for i, c in enumerate(classes):
        classes_to_channels[c] = i

    return classes_to_labels, keywords_to_custom_classes, classes_to_channels


def create_segmentation_info_json(source_folder_path:str, merged_classes:Dict[str, int], keywords_to_custom_classes:Optional[Dict[str, str]], encoded_lesions:Dict[str, int]):
    """
    Creates informative JSON files for each image, with structure as below:\n
    {\n
        "lesions": {\n
            lesion_1: [label_annotated_in_segmentation, global_label(from encoded_lesions)],\n
            lesion_2: [label_annotated_in_segmentation, global_label(from encoded_lesions)],\n
            ...\n
            lesion_n: [label_annotated_in_segmentation, global_label(from encoded_lesions)],\n
        }\n
    }

    This file contains information about custom_classes too - each of the custom classes in treated as a separate lesion.
    
    Parameters
    --------
    source_folder_path : str
        Path to the root of the data folder.
    merged_classes : Dict[str, int]
        Dictionary of classes paired with their respective integer encodings. Contains both original and custom classes.
    keywords_to_custom_classes : Dict[str, str]
        Dictionary of keywords to custom classes paired with those custom classes.
    encoded_lesions : Dict[str, int]
        Dictionary of lesions paired with integer labels.
    """
    logger = logging.getLogger("create_segmentation_info_json")
    logger.addHandler(ch)
    logger.propagate = False

    logger.info(f" Saving segmentation info...")

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.LoadPrivateTagsOn()

    for i, file in enumerate(os.listdir(os.path.join(source_folder_path, "correct_seg_nrrd"))):
        sys.stdout.write(f"\r{i+1} / {len(os.listdir(os.path.join(source_folder_path, 'correct_seg_nrrd')))}")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folder_path, "correct_seg_nrrd", file))
        seg = reader.Execute()
        dct = {"lesions" : {}}
        j = 1
        for k in seg.GetMetaDataKeys():
            if re.match(r'^Segment\d_Name$', k) and seg.GetMetaData(k) in merged_classes.keys():
                dct['lesions'][seg.GetMetaData(k)] = (j, encoded_lesions[seg.GetMetaData(k)])   # tuple(label_in_segmentation, global_label - from encoded_lesions)
                j += 1
        last_found_label_value = 0
        for key in seg.GetMetaDataKeys():
            for k in keywords_to_custom_classes.keys():
                if re.match(r'^Segment\d_Name$', key):
                    last_found_label_value += 1
                if re.findall(r"\b" + k + r"\b", seg.GetMetaData(key)):
                    dct['lesions'][keywords_to_custom_classes[k]] = (last_found_label_value, merged_classes[keywords_to_custom_classes[k]])
        json_string = json.dumps(dct)

        filename = os.path.splitext(os.path.splitext(file)[0])[0]
        with open(os.path.join(source_folder_path, "final-data", "all-checked", "seg-json", filename+".json"), 'w+') as f:
            f.write(json_string)
            f.close()
    
    print()
    logger.info(f" Segmentation info saved into JSON files.")
    
    return


def create_dataset_with_neutral_samples(source_folder_path, total_classes_count):

    logger = logging.getLogger("create_dataset_with_neutral_samples")
    logger.addHandler(ch)
    logger.propagate = False

    filenames = sorted([os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(source_folder_path, "final-data", "all-checked", "images"))])
    logger.info(f" Copying files from previous dataset...")
    l=len(filenames)
    for i, filename in enumerate(filenames):
        sys.stdout.write(f"\r{i} / {l}")
        sys.stdout.flush()

        if not os.path.exists(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "masks", filename)):
            os.mkdir(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "masks", filename))

        # firstly, copy the preprocessed data from dataset containing segmentation
        shutil.copy(os.path.join(source_folder_path, "final-data", "all-checked", "images", filename+".mha"),
                    os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "images", filename+".mha"))
        shutil.copy(os.path.join(source_folder_path, "final-data", "all-checked", "segmentations", filename+".seg.nrrd"),
                    os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "segmentations", filename+".seg.nrrd"))
        for mask in os.listdir(os.path.join(source_folder_path, "final-data", "all-checked", "masks", filename)):
            shutil.copy(os.path.join(source_folder_path, "final-data", "all-checked", "masks", filename, mask),
                        os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "masks", filename, mask))
    print()
    logger.info(f" Copying JSON files...")
    filenames = sorted([os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(source_folder_path, "final-data", "all-checked", "seg-json"))])
    l=len(filenames)
    for i, filename in enumerate(filenames):
        sys.stdout.write(f"\r{i} / {l}")
        shutil.copy(os.path.join(source_folder_path, "final-data", "all-checked", "seg-json", filename+".json"),
                    os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "seg-json", filename+".json"))
    print()
    # secondly, prepare the rest of the data - the "neutral samples", containing no segmentation
    logger.info(f" Preparing neutral samples...")
    filenames = sorted([os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(source_folder_path, "mha_no_segmentation"))])

    img_reader = sitk.ImageFileReader()
    img_reader.SetImageIO("MetaImageIO")

    img_writer = sitk.ImageFileWriter()
    img_writer.SetImageIO("NrrdImageIO")
    img_writer.UseCompressionOn()
    l=len(filenames)
    for i, filename in enumerate(filenames):
        t1 = time.time()
        if not os.path.exists(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "masks", filename)):
            os.mkdir(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "masks", filename))

        shutil.copy(os.path.join(source_folder_path, "mha_no_segmentation", filename+".mha"),
                    os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "images", filename+".mha"))
        shutil.copy(os.path.join(source_folder_path, "seg_nrrd_no_segmentation", filename+".seg.nrrd"),
                    os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "segmentations", filename+".seg.nrrd"))
        # JSON files are already prepared, so omit them here
        img_reader.SetFileName(os.path.join(source_folder_path, "mha_no_segmentation", filename+".mha"))
        img = img_reader.Execute()
        img_arr = sitk.GetArrayFromImage(img)
        for j in range(total_classes_count):
            mask = np.zeros(img_arr.shape)
            img_writer.SetFileName(os.path.join(source_folder_path, "final-data-with-neutral-samples", "all-checked", "masks", filename, "Channel"+str(j)+".seg.nrrd"))
            arr = sitk.GetImageFromArray(mask)
            arr.SetSpacing(img.GetSpacing())
            arr.SetOrigin(img.GetOrigin())
            arr.SetDirection(img.GetDirection())
            img_writer.Execute(arr)
        t2 = time.time()
        sys.stdout.write(f"\r{i+1} / {l}, previous sample time elapsed: {(t2-t1):2f} seconds")
        sys.stdout.flush()
    print()
    logger.info(" Done.")
    return

    