import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pycocotools import mask as maskUtils
from collections import defaultdict



def parse_args():
    parser = argparse.ArgumentParser(description="Create COCO format dataset from images and masks.")
    parser.add_argument('dataset_dir', type=str, help='Dataset directory containing images/train, images/val, images/test, masks/train, maasks/val, masks/test.')
    parser.add_argument('output_dir', type=str, help='Output directory for COCO annotations.')
    return parser.parse_args()

# CellPoseIDを取り出すための関数
def extract_image_name(filename):
    parts = filename.split('_')
    image_name = '_'.join(parts[:-2]) if "RoiLabel" in parts[-2] else None
    roi_label = '_'.join(parts[-2:]) if "RoiLabel" in parts[-2] else None
    return image_name, roi_label

def create_dict_ImgMask(image_dir, mask_dir):
    dict_ImgMask = defaultdict(list)
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    for mask in mask_files:

        image_name = extract_image_name(mask)[0] + '.png'
        # image_nameがimage_filesの中にあるか確認
        if image_name in image_files:
            # image_nameをキーにしてmaskをリストに追加
            dict_ImgMask[image_name].append(mask)
            
    return dict_ImgMask


def create_coco_format(dataset_dir, output_dir):
    sets = ['train', 'val', 'test']

    for dataset_type in sets:
        images_dir = os.path.join(dataset_dir, 'images', dataset_type)
        masks_dir = os.path.join(dataset_dir, 'masks', dataset_type)
        
        dict_ImgMask = create_dict_ImgMask(images_dir, masks_dir)

        coco_output = {
            "info": {
                "description": "COCO style dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "Shodai Taguchi",
                "date_created": "20240829"
            },
            "licenses": [{
                "id": 1,
                "name": "Shodai Taguchi",
                "url": ""
            }],
            "categories": [{
                "id": 1,
                "name": "PSM",
                "supercategory": ""
            }],
            "images": [],
            "annotations": []
        }

        annotation_id = 1
        for i, (image_file, mask_list) in enumerate(dict_ImgMask.items()):
            img_path = os.path.join(images_dir, image_file)

            img = cv2.imread(img_path)
            height, width, _ = img.shape
            if img.shape is None:
                print(f"Image {img_path} is None.")
            id = i + 1
            coco_output["images"].append({
                "id": id,
                "file_name": image_file,
                "width": width,
                "height": height,
                "license": 1,
                "coco_url": "",
                "flickr_url": "",
                "date_captured": ""
            })

            for j, mask in enumerate(mask_list):
                mask_path = os.path.join(masks_dir, mask)
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    segmentation = contour.flatten().tolist()

                    rle = maskUtils.encode(np.asfortranarray(mask_img))
                    rle["counts"] = rle["counts"].decode('utf-8')
                    area = float(maskUtils.area(rle))
                    bbox = cv2.boundingRect(contour)
                    x, y, w, h = bbox

                    annotation_id = str(i) + "_" + str(j)
                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "file_name": mask,
                        "image_id": id,
                        "category_id": 1,
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": [x, y, w, h],
                        "iscrowd": 0
                    })

        output_json = os.path.join(output_dir, f'{dataset_type}_annotations.json')
        with open(output_json, 'w') as f:
            json.dump(coco_output, f, indent=2)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    create_coco_format(args.dataset_dir, args.output_dir)

if __name__ == "__main__":
    main()
