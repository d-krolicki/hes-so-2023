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

logging.basicConfig(level=logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)

def extract_correct_data(source_folderpath, correct_indices):
    """
    Obsolete, not used anymore.
    """
    logger = logging.getLogger("extract_correct_data")
    logger.addHandler(ch)
    logger.propagate = False
    
    indices_len = len(correct_indices)
    nrrd_images = sorted(os.listdir(os.path.join(source_folderpath, "correct_seg_nrrd")))
    mha_images = sorted(os.listdir(os.path.join(source_folderpath, "correct_mha")))
    logger.info(f" Extracting correct data...")
    for i, index in enumerate(correct_indices):
        sys.stdout.write(f"\rCopying NRRD image {i}/{indices_len}")
        shutil.copy(os.path.join(source_folderpath, "correct_seg_nrrd", nrrd_images[index]), 
                    os.path.join(source_folderpath, "correct_seg_nrrd_filtered"))
        sys.stdout.write(f"\nCopying MHA image {i}/{indices_len}")
        sys.stdout.flush()
        shutil.copy(os.path.join(source_folderpath, "correct_mha", mha_images[index]), 
                    os.path.join(source_folderpath, "correct_mha_filtered"))
    print()
    logger.info(f" Copying finished.")


def get_empty_classes_dict(destination_folderpath, testing=False):
    logger = logging.getLogger("get_empty_classes_dict")
    logger.addHandler(ch)
    logger.propagate = False
    """
    Returns a dictionary of all existing classes (diseases) in the dataset,
    paired with their volumes set to 0.
    """
    logger.info(f" Extracting diseases in the dataset...")
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


def extract_segmentation_slices(seg:np.array):
    
    """
    Returns indices, of slices containing any segmentation in the given image. Counts from 0.
    """
    
    indices = []
    for i in range(seg.shape[0]):
        if len(np.unique(seg[i])) > 1:
            indices.append(i)
    return np.array(indices)


def extract_diseases_in_image(seg:sitk.Image):
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
    """
    i = 1

    labels_diseases = {}

    for k in seg.GetMetaDataKeys():
        if re.match(r'^Segment\d_Name$', k):
            labels_diseases[i] = seg.GetMetaData(k)
            i += 1

    return labels_diseases


def count_samples_in_classes(source_folderpath):

    logger = logging.getLogger("count_samples_in_classes")
    logger.addHandler(ch)
    logger.propagate = False

    dct = get_empty_classes_dict(os.path.join(source_folderpath, "correct_seg_nrrd"))

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")

    logger.info(f" Counting samples in classes...")
    for i, filename in enumerate(os.listdir(os.path.join(source_folderpath, "correct_seg_nrrd"))):
        sys.stdout.write(f"\r{i+1} / {len(os.listdir(os.path.join(source_folderpath, 'correct_seg_nrrd')))}")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folderpath, "correct_seg_nrrd", filename))
        image = reader.Execute()
        for k in image.GetMetaDataKeys():
            if re.match(r'^Segment\d_Name$', k):
                dct[image.GetMetaData(k).lower()] += 1
    print()
    logger.info(f" {len(os.listdir(os.path.join(source_folderpath, 'correct_seg_nrrd')))} files scanned.")
    
    return dct


def encode_lesions(source_folderpath):
    dct = {}
    for i, key in enumerate(get_empty_classes_dict(os.path.join(source_folderpath, "correct_seg_nrrd")).keys()):
        dct[key] = i
    
    return dct


def merge_classes(encoded_lesions, classes=None, custom_classes=None, keywords=None):

    logger = logging.getLogger("merge_classes")
    logger.addHandler(ch)
    logger.propagate = False

    """
    Checks if each class has more than min_samples and is suitable for training.
    Then, checks if any custom classes are passed, and if their certain criteria 
    (keywords) are provided to merge classes together according to their names.
    
    """
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


def create_segmentation_info_json(source_folderpath, merged_classes, keywords_to_custom_classes, encoded_lesions):
    
    logger = logging.getLogger("create_segmentation_info_json")
    logger.addHandler(ch)
    logger.propagate = False

    logger.info(f" Saving segmentation info...")

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.LoadPrivateTagsOn()

    for i, file in enumerate(os.listdir(os.path.join(source_folderpath, "correct_seg_nrrd"))):
        sys.stdout.write(f"\r{i+1} / {len(os.listdir(os.path.join(source_folderpath, 'correct_seg_nrrd')))}")
        sys.stdout.flush()
        reader.SetFileName(os.path.join(source_folderpath, "correct_seg_nrrd", file))
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
        with open(os.path.join(source_folderpath, "final-data", "all-checked", "seg-json", filename+".json"), 'w+') as f:
            f.write(json_string)
            f.close()
    
    print()
    logger.info(f" Segmentation info saved into JSON files.")
    
    return


def create_dataset_with_neutral_samples(source_folderpath, total_classes_count):

    logger = logging.getLogger("create_dataset_with_neutral_samples")
    logger.addHandler(ch)
    logger.propagate = False

    filenames = sorted([os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(source_folderpath, "final-data", "all-checked", "images"))])
    logger.info(f" Copying files from previous dataset...")
    l=len(filenames)
    for i, filename in enumerate(filenames):
        sys.stdout.write(f"\r{i} / {l}")
        sys.stdout.flush()

        if not os.path.exists(os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "masks", filename)):
            os.mkdir(os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "masks", filename))

        # firstly, copy the preprocessed data from dataset containing segmentation
        shutil.copy(os.path.join(source_folderpath, "final-data", "all-checked", "images", filename+".mha"),
                    os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "images", filename+".mha"))
        shutil.copy(os.path.join(source_folderpath, "final-data", "all-checked", "segmentations", filename+".seg.nrrd"),
                    os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "segmentations", filename+".seg.nrrd"))
        for mask in os.listdir(os.path.join(source_folderpath, "final-data", "all-checked", "masks", filename)):
            shutil.copy(os.path.join(source_folderpath, "final-data", "all-checked", "masks", filename, mask),
                        os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "masks", filename, mask))
    print()
    logger.info(f" Copying JSON files...")
    filenames = sorted([os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(source_folderpath, "final-data", "all-checked", "seg-json"))])
    l=len(filenames)
    for i, filename in enumerate(filenames):
        sys.stdout.write(f"\r{i} / {l}")
        shutil.copy(os.path.join(source_folderpath, "final-data", "all-checked", "seg-json", filename+".json"),
                    os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "seg-json", filename+".json"))
    print()
    # secondly, prepare the rest of the data - the "neutral samples", containing no segmentation
    logger.info(f" Preparing neutral samples...")
    filenames = sorted([os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(source_folderpath, "mha_no_segmentation"))])

    img_reader = sitk.ImageFileReader()
    img_reader.SetImageIO("MetaImageIO")

    img_writer = sitk.ImageFileWriter()
    img_writer.SetImageIO("NrrdImageIO")
    img_writer.UseCompressionOn()
    l=len(filenames)
    for i, filename in enumerate(filenames):
        t1 = time.time()
        if not os.path.exists(os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "masks", filename)):
            os.mkdir(os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "masks", filename))

        shutil.copy(os.path.join(source_folderpath, "mha_no_segmentation", filename+".mha"),
                    os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "images", filename+".mha"))
        shutil.copy(os.path.join(source_folderpath, "seg_nrrd_no_segmentation", filename+".seg.nrrd"),
                    os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "segmentations", filename+".seg.nrrd"))
        # JSON files are already prepared, so omit them here
        img_reader.SetFileName(os.path.join(source_folderpath, "mha_no_segmentation", filename+".mha"))
        img = img_reader.Execute()
        img_arr = sitk.GetArrayFromImage(img)
        for j in range(total_classes_count):
            mask = np.zeros(img_arr.shape)
            img_writer.SetFileName(os.path.join(source_folderpath, "final-data-with-neutral-samples", "all-checked", "masks", filename, "Channel"+str(j)))
            img_writer.Execute(sitk.GetImageFromArray(mask))
        t2 = time.time()
        sys.stdout.write(f"\r{i+1} / {l}, previous sample time elapsed: {(t2-t1):2f} seconds")
        sys.stdout.flush()
    print()
    logger.info(" Done.")
    return

    