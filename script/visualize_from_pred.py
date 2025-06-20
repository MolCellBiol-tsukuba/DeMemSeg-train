import os
from pathlib import Path
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm
import tifffile as tiff
import seaborn as sns

# Set2パレットをBGR変換しないRGBのままで定義
PALETTE = sns.color_palette("Set2", n_colors=10)
PALETTE = [tuple(int(c * 255) for c in rgb) for rgb in PALETTE]

def apply_masks_and_save_all(
    image_dir,
    json_dir,
    well_mask_dir,
    outdir_myvis,
    outdir_mypreds,
    score_threshold=0.5,
    alpha=0.5,
    min_area=0,
    max_area=float("inf")
):
    # wellmask_dir が None または存在しない場合はスキップ
    use_wellmask = well_mask_dir is not None and os.path.isdir(well_mask_dir)

    Path(outdir_mypreds).mkdir(parents=True, exist_ok=True)
    Path(outdir_myvis).mkdir(parents=True, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png', '.jpg', ".jpeg"))]

    for filename in tqdm(image_files, desc="Processing images"):
        name_base, _ = os.path.splitext(filename)
        image_path = Path(image_dir) / filename
        json_path = Path(json_dir) / (name_base + ".json")
        output_path_myvis = Path(outdir_myvis) / (name_base + ".png")
        output_path_mypreds = Path(outdir_mypreds) / (name_base + ".json")
        
        if not os.path.exists(json_path):
            print(f"⚠️ JSON not found for: {filename}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️ Failed to load image: {filename}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image.copy()

        # wellmask 読み込み（使う場合のみ）
        if use_wellmask:
            well_path = Path(well_mask_dir) / filename
            if not os.path.exists(well_path):
                print(f"⚠️ Wellmask not found for: {filename}")
                continue
            wellmask = tiff.imread(well_path)
            if wellmask is None:
                print(f"⚠️ Failed to load wellmask: {filename}")
                continue
            wellmask = np.where(wellmask > 0, 1, 0).astype(np.uint8)

        with open(json_path, 'r') as f:
            data = json.load(f)

        bboxes = data["bboxes"]
        masks = data["masks"]
        scores = data["scores"]
        labels = data.get("labels", [0] * len(bboxes))

        filtered_bboxes = []
        filtered_scores = []
        filtered_masks = []
        filtered_labels = []
        filtered_labelIDs = []

        label_ID = 1
        for bbox, score, rle, label in zip(bboxes, scores, masks, labels):
            if score < score_threshold:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            area = w * h

            if area < min_area or area > max_area:
                continue

            mask = maskUtils.decode(rle).astype(np.uint8)

            # wellmaskフィルタリング（使用時のみ）
            if use_wellmask:
                intersection = np.logical_and(mask, wellmask)
                intersection_area = np.sum(intersection)
                mask_area = np.sum(mask)
                if mask_area == 0 or (intersection_area / mask_area) < 0.1:
                    continue

            # 描画
            filtered_bboxes.append(bbox)
            filtered_scores.append(score)
            filtered_masks.append(rle)
            filtered_labels.append(label)
            filtered_labelIDs.append(label_ID)

            n_colors = len(PALETTE)
            color = PALETTE[label_ID % n_colors]

            for c in range(3):
                overlay[:, :, c] = np.where(mask == 1,
                                            overlay[:, :, c] * (1 - alpha) + alpha * color[c],
                                            overlay[:, :, c])
            label_ID += 1

        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path_myvis), overlay_bgr)

        filtered_data = {
            "labelIDs": filtered_labelIDs,
            "labels": filtered_labels,
            "scores": filtered_scores,
            "bboxes": filtered_bboxes,
            "masks": filtered_masks
        }
        with open(output_path_mypreds, 'w') as f:
            json.dump(filtered_data, f)

    print(f"\n✅ All overlays saved to: {outdir_myvis}")
    print(f"✅ All filtered JSONs saved to: {outdir_mypreds}")


# 実行例（必要に応じてパスを変更）
if __name__ == "__main__":
    image_dir = "path/to/image_dir"
    json_dir = "path/to/json_dir"
    well_mask_dir = "path/to/well_mask_dir"
    outdir_myvis = "path/to/output_vis_dir"
    outdir_mypreds = "path/to/output_preds_dir"
    score_threshold = 0.5
    alpha = 0.5
    min_area = 0
    max_area = float("inf")