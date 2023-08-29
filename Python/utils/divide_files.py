import sys
sys.path.append("/OLD-DATA-STOR/HESSO_Internship_2023/Dariusz")
import os
import shutil
import SimpleITK as sitk
import re
import logging
import csv

logging.basicConfig(level=logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)

def extract_correct_files(source_folderpath):

    logger = logging.getLogger("extract_correct_files")
    logger.addHandler(ch)
    logger.propagate = False

    """
    Extract corresponding .MHA and .SEG.NRRD files and place them in their corresponding folders.
    """
    cnt_correct_mha = 0
    cnt_correct_seg = 0
    logger.info(" Extracting correct SEG and MHA files...")
    ldir = os.listdir(source_folderpath)
    for i, filename in enumerate(ldir):
        sys.stdout.write(f"\r{i+1} / {len(os.listdir(source_folderpath))}")
        sys.stdout.flush()
        if not filename.startswith('.') and filename.endswith('.mha') and os.path.splitext(filename)[0]+".seg.nrrd" in ldir:
            shutil.copy(os.path.join(source_folderpath, filename),
                        os.path.join(source_folderpath, "correct_mha" ,filename))
            cnt_correct_mha += 1
        elif not filename.startswith('.') and filename.endswith('.seg.nrrd') and os.path.splitext(os.path.splitext(filename)[0])[0]+".mha" in ldir:
            shutil.copy(os.path.join(source_folderpath, filename),
                        os.path.join(source_folderpath, "correct_seg_nrrd" ,filename))
            cnt_correct_seg += 1
    print()
    logger.info(f" {i+1} files scanned.")
    logger.info(f" {cnt_correct_mha} MHA files found. {cnt_correct_seg} SEG files found.")

    corrupted_files = 0

    logger.info(f" Checking for any corrupted files...")
    l = len(os.listdir(os.path.join(source_folderpath, 'correct_seg_nrrd')))
    with open("Results/corrupted_files.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        reader = sitk.ImageFileReader()

        reader.SetImageIO("MetaImageIO")
        for i, filename in enumerate(os.listdir(os.path.join(source_folderpath, "correct_mha"))):
            sys.stdout.write(f"\r{i+1} / {l}")
            sys.stdout.flush()
            filename_noext = os.path.splitext(filename)[0]
            try:
                reader.SetFileName(os.path.join(source_folderpath, "correct_mha", filename))
                reader.Execute()
            except:
                print()
                logger.warning(f" Encountered problem with file {filename}.")
                logger.warning(f" It is most likely due to file corruption. The file was omitted and removed from the dir.")
                logger.warning(f" Proceeding with further files...")
                os.remove(os.path.join(source_folderpath, "correct_mha", filename))
                os.remove(os.path.join(source_folderpath, "correct_seg_nrrd", filename_noext+".seg.nrrd"))
                writer.writerow([filename_noext])
                corrupted_files += 1

        reader.SetImageIO("NrrdImageIO")
        for i, filename in enumerate(os.listdir(os.path.join(source_folderpath, "correct_seg_nrrd"))):
            sys.stdout.write(f"\r{i+1} / {l}")
            sys.stdout.flush()
            filename_noext = os.path.splitext(os.path.splitext(filename)[0])[0]
            try:
                reader.SetFileName(os.path.join(source_folderpath, "correct_seg_nrrd", filename))
                reader.Execute()
            except:
                print()
                logger.warning(f" Encountered problem with file {filename}.")
                logger.warning(f" It is most likely due to file corruption. The file was omitted and removed from the dir.")
                logger.warning(f" Proceeding with further files...")
                os.remove(os.path.join(source_folderpath, "correct_mha", filename_noext+".mha"))
                os.remove(os.path.join(source_folderpath, "correct_seg_nrrd", filename))
                writer.writerow([filename_noext])
                corrupted_files += 1
                
    print()
    logger.info(f" {corrupted_files} corrupted files found and removed.")
    return 0


def extract_files_without_segmentation(source_folderpath):

    logger = logging.getLogger("extract_files_without_segmentation")
    logger.addHandler(ch)
    logger.propagate = False

    """
    Separate files with complete and incomplete segmentation, and place them in their corresponding folders.\n
    Segmentation complete: correct_mha_filtered, correct_seg_nrrd_filtered.\n
    No segmentation: mha_no_segmentation, seg_nrrd_no_segmentation.
    """
    logger.info(f" Extracting files without segmentation...")
    filenames = os.listdir(os.path.join(source_folderpath, "correct_seg_nrrd"))
    with open("Results/corrupted_files.csv", "w") as f:
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        for i, filename in enumerate(filenames):
            sys.stdout.write(f"\r{i+1} / {len(filenames)}")
            sys.stdout.flush()

            reader.SetFileName(os.path.join(source_folderpath, "correct_seg_nrrd", filename))
            image = reader.Execute()

            segmentation_found = False

            for key in image.GetMetaDataKeys():
                if re.match(r'^Segment\d_Name$', key):
                    segmentation_found = True
            
            if segmentation_found and image.GetSize()[0] > 1:
                shutil.copy(os.path.join(source_folderpath, "correct_seg_nrrd", filename),
                            os.path.join(source_folderpath, "correct_seg_nrrd_filtered"))
                shutil.copy(os.path.join(source_folderpath, "correct_mha", os.path.splitext(os.path.splitext(filename)[0])[0]+".mha"),
                            os.path.join(source_folderpath, "correct_mha_filtered"))
            else:
                shutil.copy(os.path.join(source_folderpath, "correct_seg_nrrd", filename),
                            os.path.join(source_folderpath, "seg_nrrd_no_segmentation"))
                shutil.copy(os.path.join(source_folderpath, "correct_mha", os.path.splitext(os.path.splitext(filename)[0])[0]+".mha"),
                            os.path.join(source_folderpath, "mha_no_segmentation"))

    f.close()            
    print()
    logger.info(f" {i+1} files scanned.")
    logger.info(f" Found {len(os.listdir(os.path.join(source_folderpath, 'correct_mha_filtered')))} MHA files.")
    logger.info(f" Found {len(os.listdir(os.path.join(source_folderpath, 'correct_seg_nrrd_filtered')))} SEG files.")
    return 0


