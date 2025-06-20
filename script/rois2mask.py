import os
import subprocess
import tifffile
import numpy as np
from read_roi import read_roi_zip
from skimage.draw import polygon


# 親ディレクトリから再帰的にファイルを探す
def find_all_file(parent_dir, sub_dir, filename):
    all_files = []
        
    # 親ディレクトリを再帰的に歩く
    for root, dirs, files in os.walk(parent_dir):
        # 現在のディレクトリ内のファイルを探す
        if sub_dir in root:
            if filename in files:
                # 見つかった場合、絶対パスをリストに追加
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
                        # "Labels"の値を取得
                        crop_files = tag.value["Labels"]
    return crop_files

def roikey_filename_dict(rois, crop_files):
    # roisのkeyのリストを取得
    roi_keys = list(rois.keys())
    # 最初の-までの文字列を抜き出す
    roi_keys = [key.split("-")[0] for key in roi_keys]
    # 重複を消す
    roi_keys = list(set(roi_keys))
    roi_keys = sorted(roi_keys)
    # roiのkeyとcrop_filesを対応させるためにdictを作成。keyはroiのkey, valueはcrop_files
    # roi_dict = {key: value for key, value in zip(roi_keys, crop_files)}
    
    # roi_keysの要素を数字に変換して、crop_filesのindexと対応させる
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
    
# directoryの中を空にする
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
    
    # parent dirの取得
    parent_dir = os.path.dirname(os.path.dirname(target_stack_path))
    
    # mask_imageディレクトリがなければ作成
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
    
    # roiをmaskとしてスライス画像のファイル名を用いて保存するために、stack画像のスライスそれぞれのファイル名を取得する。
    crop_files = filename_list(path_stack)
    
    # crop_filesとroi_keysを対応させるためのdictを作成
    roi_dict = roikey_filename_dict(rois, crop_files)
    
    #　roi_dictのvalueを取得し、それをファイル名としマスク画像を保存する
    height, width = 200, 200
    
    for key, value in rois.items():
        x = rois[key]['x']
        y = rois[key]['y']
        mask = np.zeros((height, width), dtype=np.uint8)
        rr, cc = polygon(y, x)
        mask[rr, cc] = 255
        


        # maskの保存、keyの-までを取得し、roi_dictのvalueを取得し、それをファイル名とする
        key_in_roi_dict = key.split("-")[0]
        key2file = roi_dict.get(key_in_roi_dict)
        file_name = key2file.split(".png")[0]
        roi_label = str(key)
        mask_path = os.path.join(path_mask, strain_name + "_" + file_name + "_RoiLabel_" + roi_label + ".png")
        tifffile.imwrite(mask_path, mask)
        
        # train_imageのコピー
        extract_image_path = os.path.join(path_extract, key2file)
        train_path = os.path.join(path_train, strain_name + "_" + file_name + ".png") 
        cp_image(extract_image_path, train_path)