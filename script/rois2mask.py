import os
import subprocess
import tifffile
import numpy as np
from read_roi import read_roi_zip
from skimage.draw import polygon


# Recursively search for target files starting from the parent directory
def find_all_file(parent_dir, sub_dir, filename):
    all_files = []
        
    # Walk through the directory tree
    for root, dirs, files in os.walk(parent_dir):
        # Inspect directories that include the target sub-directory fragment
        if sub_dir in root:
            if filename in files:
                # Store the absolute path for any match
                all_files.append(os.path.join(root, filename))
            else:
                print(f"{filename} not found in {root}")

    return all_files


def filename_list(path_stack):
    crop_files = []
    with tifffile.TiffFile(path_stack) as tif:
        for page in tif.pages:
            for tag in page.tags:
                if tag.name == "IJMetadata":
                    if "Labels" in page.tags["IJMetadata"].value:
                        # Retrieve the list stored under "Labels"
                        crop_files = tag.value["Labels"]
    return crop_files

def roikey_filename_dict(rois, crop_files):
    # Collect the ROI keys
    roi_keys = list(rois.keys())
    # Keep the prefix before the first hyphen
    roi_keys = [key.split("-")[0] for key in roi_keys]
    # Remove duplicates and sort numerically
    roi_keys = list(set(roi_keys))
    roi_keys = sorted(roi_keys)
    # Build a mapping between ROI identifiers and crop file names
    # roi_dict = {key: value for key, value in zip(roi_keys, crop_files)}
    
    # Convert the ROI identifiers to match the index of the crop file list
    roi_dict = {}
    for index in range((len(roi_keys))): 
        int_index = int(roi_keys[index]) -1
        roi_dict[roi_keys[index]] = crop_files[int_index]
        
    return roi_dict

    
def cp_image(file_path, out_path):
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    subprocess.run(["cp", file_path, out_path])
    
# Remove every file and sub-directory inside the target directory
def clean_directory(dir):
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clean_directory(file_path)
            os.rmdir(file_path)


def rois2mask(target_stack_path, sub_dir, stack_file):
    rois = read_roi_zip(target_stack_path)
    
    # Retrieve the parent directory
    parent_dir = os.path.dirname(os.path.dirname(target_stack_path))
    
    # Create the mask_image directory if needed
    path_extract = os.path.join(parent_dir, "extract_image")
    # get absolute path of the extract image
    abs_path_extract = os.path.abspath(path_extract)
    strain_name = abs_path_extract.split("/")[-4]
    
    path_mask = os.path.join(parent_dir, "mask_image")
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)      
    clean_directory(path_mask)
    
    path_train = os.path.join(parent_dir, "train_image")
    if not os.path.exists(path_train):
        os.makedirs(path_train)      
    clean_directory(path_train)
    
    path_stack = os.path.join(parent_dir, sub_dir, stack_file)
    
    # Read the slice filenames so we can reuse them when saving masks
    crop_files = filename_list(path_stack)
    
    # Build a dictionary that links ROI keys to crop filenames
    roi_dict = roikey_filename_dict(rois, crop_files)
    
    # Use the ROI metadata to set a consistent output filename
    height, width = 200, 200
    
    for key, value in rois.items():
        x = rois[key]['x']
        y = rois[key]['y']
        mask = np.zeros((height, width), dtype=np.uint8)
        rr, cc = polygon(y, x)
        mask[rr, cc] = 255
        


        # Save the mask with the strain and ROI identifier embedded in the filename
        key_in_roi_dict = key.split("-")[0]
        key2file = roi_dict.get(key_in_roi_dict)
        file_name = key2file.split(".png")[0]
        roi_label = str(key)
        mask_path = os.path.join(path_mask, strain_name + "_" + file_name + "_RoiLabel_" + roi_label + ".png")
        tifffile.imwrite(mask_path, mask)
        
        # Copy the corresponding crop into the training image directory
        extract_image_path = os.path.join(path_extract, key2file)
        train_path = os.path.join(path_train, strain_name + "_" + file_name + ".png") 
        cp_image(extract_image_path, train_path)
