import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")

from Python.utils.utils import train_test_val_split

train_test_val_split(
    image_dir="/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz/Data/COMPLETE_DATA/all_classes/unified_resolution/images",
    csv_folder="/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz/Data/COMPLETE_DATA/all_classes/unified_resolution/csv",
    val_size=0.2,
    test_size=0.0
)
