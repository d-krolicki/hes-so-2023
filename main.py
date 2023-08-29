from Python.utils import utils, preprocessing, divide_files
import logging
import time
# 
# tim1 = time.time()

SOURCE_FOLDERPATH = "/OLD-DATA-STOR/segmentation ovud"

IMAGE_DIR = "/OLD-DATA-STOR/segmentation ovud/final-data-with-neutral-samples/all-checked/images"
LABEL_DIR = "/OLD-DATA-STOR/segmentation ovud/final-data-with-neutral-samples/all-checked/masks"
CSV_DIR = "/OLD-DATA-STOR/segmentation ovud/final-data-with-neutral-samples/all-checked/csv"

CLASSES = ['axillary lymph node', 'sternal lymph node', 'hepatic mass',
                                                        'lung consolidation', 'lipoma', 'subcutaneous mass',
                                                        'lung mass', 'interstitial pattern']

testing = False
if testing:
    logging.warning(f"The program is operating in testing mode on a subset of the data.")
    logging.warning(f"To exit testing mode, please set TESTING flag to False.")



# Step 0
# divide_files.extract_correct_files(SOURCE_FOLDERPATH)

# divide_files.extract_files_without_segmentation(SOURCE_FOLDERPATH)

# Step 1
# utils.correct_typos_in_lesions_names(SOURCE_FOLDERPATH, CLASSES)

# Step 2
# sample_counts = utils.count_samples_in_classes(SOURCE_FOLDERPATH)


# Step 3
encoded_lesions = utils.encode_lesions(SOURCE_FOLDERPATH)


merged_classes, keywords_to_custom_classes, classes_to_channels = utils.merge_classes(encoded_lesions = encoded_lesions,
                                                                classes = CLASSES,
                                                                custom_classes = ['unidentified node'],
                                                                keywords = ['node'])
print("=================================")
print("======== Merged classes: ========")
print("=================================")
for k, v in merged_classes.items():
    print(f"{k} : {v}")

print()
print()
print()
print(f"Encoded lesions:")
for k, v in encoded_lesions.items():
    print(f"{k} : {v}")

# Step 4

# utils.create_segmentation_info_json(SOURCE_FOLDERPATH, merged_classes, keywords_to_custom_classes, encoded_lesions)

# Step 5

# preprocessing.check_metadata_consistency(SOURCE_FOLDERPATH)


# Step 6

# preprocessing.prepare_segmentation_dirs(SOURCE_FOLDERPATH)

# preprocessing.prepare_segmentation_masks(SOURCE_FOLDERPATH,
#                                          classes_to_channels,
#                                          custom_classes = ['unidentified node'])

# # Step 7

# utils.create_dataset_with_neutral_samples(SOURCE_FOLDERPATH, len(merged_classes))

# preprocessing.unify_spacings(SOURCE_FOLDERPATH)
# # Step 8 - optional, if necessary

# utils.train_test_val_split(image_dir=IMAGE_DIR, label_dir=LABEL_DIR, csv_folder=CSV_DIR)
# tim2 = time.time()