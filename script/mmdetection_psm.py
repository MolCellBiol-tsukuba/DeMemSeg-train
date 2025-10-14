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


## Convert RLE masks stored in the JSON files into pickle-style arrays
pred_dir = os.path.join(path_outdir, "preds")
list_json = glob.glob(f'{pred_dir}/*.json')

# Keep entries whose scores exceed the configured threshold
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


# Prepare to write shape descriptors to CSV
path_csv = os.path.join(path_outdir, "shape_masks.csv")

header = ['JsonPath','ImageInfPath', 'ImagePath', 'ImageName', 'MaskIndex','Score', 'Area', 'Intensity', 'Ratio', 'Perimeter', 'Compactness', 'AspectRatio', 'Circularity', 'Roundness', 'CentroidX', 'CentroidY'] + \
         [f'HuMoment_{i+1}' for i in range(7)] + ['FourierDescriptor1', 'FourierDescriptor2', 'FourierDescriptor3', 'FourierDescriptor4', 'FourierDescriptor5']
        
# Create an empty CSV with headers
with open(path_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    
    
    
num_masks = len(data["masks"])

# Process each mask candidate
for mask_idx in range(num_masks):
    # Resolve the JSON path associated with the current mask
    json_path = data["path"][mask_idx]
    image_name = os.path.basename(json_path).replace('.json', '.png')
    print(f'Processing {image_name}...')
    
    # Retrieve the detection score
    score = data["scores"][mask_idx]
    
    # Locate the rendered visualization that MMDetection saved
    vis_dir = os.path.join(path_outdir, "vis")
    image_inf_path = os.path.join(vis_dir, image_name)
    
    # Locate the original pre-inference image
    image_path = os.path.join(path_indir, image_name)

    # Decode the binary mask
    binary_image = mask.decode([data["masks"][mask_idx]])

    # Extract contours from the decoded mask
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 1. Area
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
        
        # 2. Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # 3. Compactness = perimeter^2 / area
        if not area == 0:
            compactness = (perimeter ** 2) / area
        else:
            compactness = 0
       
        
        # 4. Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # 5. Circularity = 4 * pi * area / perimeter^2
        if not perimeter == 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            # circularity = 0
            continue
            
        # 6. Major axis length based on the minimum-area rectangle
        rect = cv2.minAreaRect(contour)  # Minimum bounding rotated rectangle
        width, height = rect[1]  # Extract width and height
        major_axis = max(width, height)  # Treat the longer side as the major axis

        if major_axis > 0:  # Only compute roundness if the major axis is non-zero
            roundness = (4 * area) / (np.pi * (major_axis ** 2))
        else:
            roundness = 0  # Leave as zero to avoid division by zero
            
        # 7. Centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        # 8. Hu moments
        hu_moments = cv2.HuMoments(M).flatten()
        
        # 9. Fourier descriptors
        contour_complex = np.empty(contour.shape[0], dtype=complex)  # Allocate array sized to the contour points
        contour_complex.real = contour[:, 0, 0]  # Store x coordinates in the real part
        contour_complex.imag = contour[:, 0, 1]  # Store y coordinates in the imaginary part
        fourier_descriptors = fft(contour_complex)

        # Extract the first five Fourier descriptors (tune as needed)
        fourier_descriptors_real = fourier_descriptors[:5].real

        # Append the computed descriptors to the CSV file
        row = [json_path, image_inf_path, image_path, image_name, mask_idx, score, 
               area, mean_intensity, ratio, perimeter, compactness, aspect_ratio, circularity, roundness, cx, cy] + \
              list(hu_moments) + list(fourier_descriptors_real)

        with open(path_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
