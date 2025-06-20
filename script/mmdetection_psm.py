from mmdet.apis import DetInferencer
import glob
import os
import json
import pickle
import pycocotools.mask as mask
import cv2
import numpy as np
from scipy.fft import fft
import csv
import sys

# input
args = sys.argv

path_indir = args[1]

path_MMDet = args[2]
path_outdir = os.path.join(os.path.dirname(path_indir), path_MMDet)
os.makedirs(path_outdir, exist_ok=True)


model = args[3]
weights = args[4]

inferencer = DetInferencer(model=model, weights=weights, device='cuda:0')

score_threshold = 0.3

inferencer(inputs=path_indir,
           out_dir=path_outdir, 
           pred_score_thr=score_threshold,
           no_save_pred=False,
           print_result=False)


## json fileにあるRLE形式のmaskをpkl形式に変換
pred_dir = os.path.join(path_outdir, "preds")
list_json = glob.glob(f'{pred_dir}/*.json')

# scoreが0.85以上のものをpklとして保存
filtered_data = {
    'path': [],
    'labels': [],
    'scores': [],
    'bboxes': [],
    'masks': [],
    'scores': []
}

for _json in list_json:
    json_open = open(_json)
    json_load = json.load(json_open)
    
    for i, score in enumerate(json_load['scores']):
        if score >= score_threshold:
            filtered_data['path'].append(_json)
            filtered_data['labels'].append(json_load['labels'][i])
            filtered_data['scores'].append(json_load['scores'][i])
            filtered_data['bboxes'].append(json_load['bboxes'][i])
            filtered_data['masks'].append(json_load['masks'][i])
            filtered_data['scores'].append(json_load['scores'][i])
    
    
data = filtered_data
maskedArr = mask.decode([data["masks"][0]])


# CSVファイルに書き込む準備
path_csv = os.path.join(path_outdir, "shape_masks.csv")

header = ['JsonPath','ImageInfPath', 'ImagePath', 'ImageName', 'MaskIndex','Score', 'Area', 'Intensity', 'Ratio', 'Perimeter', 'Compactness', 'AspectRatio', 'Circularity', 'Roundness', 'CentroidX', 'CentroidY'] + \
         [f'HuMoment_{i+1}' for i in range(7)] + ['FourierDescriptor1', 'FourierDescriptor2', 'FourierDescriptor3', 'FourierDescriptor4', 'FourierDescriptor5']
        
# CSVファイルを新規作成し、ヘッダーを書き込む
with open(path_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    
    
    
num_masks = len(data["masks"])

# 各マスクについて処理
for mask_idx in range(num_masks):
    # json_pathを取得
    json_path = data["path"][mask_idx]
    image_name = os.path.basename(json_path).replace('.json', '.png')
    print(f'Processing {image_name}...')
    
    # 予測のスコアを取得
    score = data["scores"][mask_idx]
    
    # 予測画像の絶対パスを取得。2階層上にあるvisディレクトリに保存されている
    vis_dir = os.path.join(path_outdir, "vis")
    image_inf_path = os.path.join(vis_dir, image_name)
    
    # 予測前画像の絶対パスを取得
    image_path = os.path.join(path_indir, image_name)

    # バイナリ形式のマスクをデコード
    binary_image = mask.decode([data["masks"][mask_idx]])

    # 輪郭を抽出
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 1. 面積
        area = cv2.contourArea(contour)
        
        # 2. Intensity (Mean Intensity) that is analyse for image_path
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask_intensity = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask_intensity, [contour], -1, 255, -1)
        mean_intensity = cv2.mean(image, mask=mask_intensity)[0]
        
        # 3. Ratio (Area / Intensity)
        if not mean_intensity == 0:
            ratio = area / mean_intensity
        else:
            ratio = 0
        
        # 2. 周囲長 (Perimeter)
        perimeter = cv2.arcLength(contour, True)
        
        # 3. コンパクトさ (Compactness) = 周囲長^2 / 面積
        if not area == 0:
            compactness = (perimeter ** 2) / area
        else:
            compactness = 0
       
        
        # 4. アスペクト比 (Aspect Ratio)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # 5. 円形度 (Circularity) = 4π × 面積 / 周囲長^2
        if not perimeter == 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            # circularity = 0
            continue
            
        # 6. 長軸 (Major Axis)
        rect = cv2.minAreaRect(contour)  # 最小外接回転矩形
        width, height = rect[1]  # 幅と高さを取得
        major_axis = max(width, height)  # 長軸

        if major_axis > 0:  # 長軸が0でないときにRoundnessを計算
            roundness = (4 * area) / (np.pi * (major_axis ** 2))
        else:
            roundness = 0  # 0のままにしておく
            
        # 7. 重心 (Centroid)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        # 8. Huモーメント
        hu_moments = cv2.HuMoments(M).flatten()
        
        # 9. フーリエ記述子 (Fourier Descriptors)
        contour_complex = np.empty(contour.shape[0], dtype=complex)  # 輪郭点の数だけの配列を用意
        contour_complex.real = contour[:, 0, 0]  # x座標を実部に
        contour_complex.imag = contour[:, 0, 1]  # y座標を虚部に
        fourier_descriptors = fft(contour_complex)

        # 5つのフーリエ記述子を抽出（適宜調整可能）
        fourier_descriptors_real = fourier_descriptors[:5].real

        # データをCSVファイルに書き込む
        row = [json_path, image_inf_path, image_path, image_name, mask_idx, score, 
               area, mean_intensity, ratio, perimeter, compactness, aspect_ratio, circularity, roundness, cx, cy] + \
              list(hu_moments) + list(fourier_descriptors_real)

        with open(path_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)