import os
from cellpose import models, io, plot
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import sys


# input
args = sys.argv

path_indir = args[1]
path_CPout = args[2]
prefix = args[3]
bf_or_fluo = args[4]
path_to_indir = os.path.join(path_indir)
print(path_to_indir)

crop_size = 100
BF_AllinOne = False
Merge_AllinOne = True
Ch00_AllinOne = False
Ch01_AllinOne = True
Ch02_AllinOne = False


path_to_outdir = os.path.join(path_to_indir, path_CPout)
if not os.path.exists(path_to_outdir):
    os.makedirs(path_to_outdir)


def crop_roi(path_img, path_mask, path_outdir, size=100, AllinOne=False):
    #画像の読み込み
    cv2_img = cv2.imread(path_img)
    cv2_mask = path_mask
    lower_limit = 1
    upper_limit = int(np.max(cv2_mask) + 1)
    
    path_outdir_extract = os.path.join(path_outdir, "extract_image")
    os.makedirs(path_outdir_extract, exist_ok=True)
    
    path_outdir_gray = os.path.join(path_outdir, "gray_image")
    os.makedirs(path_outdir_gray, exist_ok=True)
    
    path_outdir_crop = os.path.join(path_outdir, "crop_image")
    os.makedirs(path_outdir_crop, exist_ok=True)
    
    print("===", path_img, "===")
    
    for threshold in tqdm(range(lower_limit, upper_limit)):
        mask = cv2.inRange(cv2_mask, threshold, threshold)
        if mask.sum() == 0:
            continue
        else:
            binary_image = mask.astype(np.uint8)
            
            # マスク画像の輪郭を修正
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            # dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
            # closed_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel)
            
            cv2_img_extract = cv2.extract_and(cv2_img, cv2_img, mask=binary_image)         
            
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for index, contour in enumerate(contours):
                # # ROIを作成
                # x, y, w, h = cv2.boundingRect(contour)
                # roi = cv2_img[y:y+h, x:x+w]

                # ROIの重心を計算
                # 輪郭が存在するか確認
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue    
                else:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                # 重心から+/- 100 ピクセルの正方形領域を切り抜く
                size = size
                height, width, _ = cv2_img_extract.shape
                
                # CellPoseでsegmentした領域でcropした画像を保存する
                crop_region_extract = cv2_img_extract[max(0, cy-size):min(cy+size, height), max(0, cx-size):min(cx+size, width)]
                basename_img = os.path.basename(path_img).split(".")[0]
                output_path = "crop_"+ str(basename_img) +"_"+ str(threshold) +"_"+ str(index) + "_crop_region_extract.png"
                cv2.imwrite(os.path.join(path_outdir_extract, output_path), crop_region_extract)
                # 一つのフォルダにまとめたければ
                if AllinOne:
                    name_ch = basename_img.split("_")[-1]
                    name_dir = "All_extract_" + name_ch
                    path_sum_outdir = os.path.join(os.path.dirname(path_outdir), name_dir)
                    if not os.path.exists(path_sum_outdir):
                        os.makedirs(path_sum_outdir)
                    
                    cv2.imwrite(os.path.join(path_sum_outdir, output_path), crop_region_extract)
                
                # gray scaleの画像も保存する
                cv2_img_gray_extract = cv2.cvtColor(cv2_img_extract, cv2.COLOR_BGR2GRAY)
                crop_region_gray = cv2_img_gray_extract[max(0, cy-size):min(cy+size, height), max(0, cx-size):min(cx+size, width)]
                output_path_gray = "crop_"+ str(basename_img) +"_"+ str(threshold) +"_"+ str(index) + "_crop_region_gray.png"
                cv2.imwrite(os.path.join(path_outdir_gray, output_path_gray), crop_region_gray)
                
                # cropした画像を保存する
                crop_region = cv2_img[max(0, cy-size):min(cy+size, height), max(0, cx-size):min(cx+size, width)]
                output_path = "crop_"+ str(basename_img) +"_"+ str(threshold) +"_"+ str(index) + "_crop_region.png"
                cv2.imwrite(os.path.join(path_outdir_crop, output_path), crop_region)
                


def pred_result(path_to_indir, prefix="Series*_Processed001_ch00.tif", bf_or_fluo="FLUO"):
    if bf_or_fluo == "BF":
        # 以下のモデルはＢＦ画像で細胞をセグメンテーションするためのモデル
        model_path = "../model/CP_20250109_172215_BrightField_LearningRate01_WeightDecay00001_Nepoch10000"
    elif bf_or_fluo == "FLUO":
        # 以下のモデルはＰＳＭの蛍光画像で細胞をセグメンテーションするためのモデル
        model_path = "../model/CP_20241010_162052_mCh20_LearningRate01_WeightDecay00001_Nepoch10000"
    else:
        print("Please select BF or FLUO")
        sys.exit()
        
    model = models.CellposeModel(pretrained_model=model_path, gpu=True)
    imgs = []
    path_to_indir = path_to_indir
    list_of_imgs = glob(os.path.join(path_to_indir, prefix))
    list_of_imgs.sort()
    num_of_imgs = len(list_of_imgs)

    for img_path in list_of_imgs:
        imgs.append(io.imread(filename=img_path))
        
        
    result = model.eval(x=imgs,
                        diameter=135,
                        channels=[0, 0],)
    
    return result, imgs, list_of_imgs, num_of_imgs
    
    
def save_result(result, imgs, list_of_imgs, num_of_imgs):
    for i in tqdm(range(num_of_imgs)):
        file_name = os.path.basename(list_of_imgs[i]).split(".")[0]
        io.save_masks(images=imgs[i], masks=result[0][i],
                    flows=result[1][i], 
                    file_names = os.path.join(path_to_outdir, file_name),
                    png=True
                    )
        # save flow fig
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img=imgs[i], 
                               maski=result[0][i], 
                               flowi=result[1][i][0], 
                               file_name=os.path.join(path_to_outdir, file_name)
        )
        # io.save_rois(masks=result[0][i],
        #             file_name = os.path.join(path_to_outdir, file_name)
        # ) 

result, imgs, list_of_imgs, num_of_imgs = pred_result(path_to_indir, prefix)
save_result(result, imgs, list_of_imgs, num_of_imgs)
    

def get_path_list_from_out(common_name, path_to_outdir=path_to_outdir):
    common_name = common_name
    list_of_imgs = glob(os.path.join(path_to_outdir, common_name))
    list_of_imgs.sort()
    return list_of_imgs

def get_path_list_from_in(common_name, path_to_indir=path_to_indir):
    common_name = common_name
    list_of_imgs = glob(os.path.join(path_to_indir, common_name))
    list_of_imgs.sort()
    return list_of_imgs

mask = "*_cp_masks.png"
list_of_masks = get_path_list_from_out(mask)

bf = "Image*.tif"
list_of_bfs = get_path_list_from_in(bf)

merge = "Series*_Processed001.tif"
list_of_merges = get_path_list_from_in(merge)

ch00 = "Series*_Processed001_ch00.tif"
list_of_ch00 = get_path_list_from_in(ch00)

ch01 = "Series*_Processed001_ch01.tif"
list_of_ch01 = get_path_list_from_in(ch01)

ch02 = "Series*_Processed001_ch02.tif"
list_of_ch02 = get_path_list_from_in(ch02)


for index, i_mask in enumerate(list_of_masks):
    # cv2を使ってマスク画像を読み込む
    mask_array = cv2.imread(i_mask, cv2.IMREAD_UNCHANGED)
    
    # まずはbfの画像からROIを切り抜く
    if len(list_of_bfs) != 0:
        path_img = list_of_bfs[index]
        name_outdir = os.path.basename(path_img).split(".")[0]
        path_outdir = os.path.join(path_to_outdir, "crop_" + name_outdir)    
        crop_roi(path_img, mask_array, path_outdir, size=crop_size, AllinOne=BF_AllinOne)
    
    # mergeの画像からROIを切り抜く
    if len(list_of_merges) != 0:
        path_img = list_of_merges[index]
        name_outdir = os.path.basename(path_img).split(".")[0]
        path_outdir = os.path.join(path_to_outdir, "crop_" + name_outdir)
        crop_roi(path_img, mask_array, path_outdir, size=crop_size, AllinOne=Merge_AllinOne)

    # ch00の画像からROIを切り抜く
    if len(list_of_ch00) != 0:
        path_img = list_of_ch00[index]
        name_outdir = os.path.basename(path_img).split(".")[0]
        path_outdir = os.path.join(path_to_outdir, "crop_" + name_outdir)
        crop_roi(path_img, mask_array, path_outdir, size=crop_size, AllinOne=Ch00_AllinOne)
    
    # ch01の画像からROIを切り抜く
    if len(list_of_ch01) != 0:        
        path_img = list_of_ch01[index]
        name_outdir = os.path.basename(path_img).split(".")[0]
        path_outdir = os.path.join(path_to_outdir, "crop_" + name_outdir)
        crop_roi(path_img, mask_array, path_outdir, size=crop_size, AllinOne=Ch01_AllinOne)

    # ch02の画像からROIを切り抜く
    if len(list_of_ch02) != 0:        
        path_img = list_of_ch02[index]
        name_outdir = os.path.basename(path_img).split(".")[0]
        path_outdir = os.path.join(path_to_outdir, "crop_" + name_outdir)
        crop_roi(path_img, mask_array, path_outdir, size=crop_size, AllinOne=Ch02_AllinOne)
