import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm

DATASET_ROOT = "./train-2/Scenario-A"
OUTPUT_ROOT  = "./sceneA_degradado"

TARGET_SIZE_HR = (128, 64)
TARGET_SIZE_LR = (128, 64)

SEED = 42
VAL_RATIO = 0.10

def get_corners_from_json(corners_dict, img_key):
    c = corners_dict[img_key]
    return np.array([
        c["top-left"],
        c["top-right"],
        c["bottom-right"],
        c["bottom-left"]
    ], dtype=np.float32)

def crop_and_warp(img, corners, target_size):
    w, h = target_size
    dst_pts = np.array([
        [0,   0  ],
        [w-1, 0  ],
        [w-1, h-1],
        [0,   h-1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    return cv2.warpPerspective(img, M, (w, h))

def coletar_tracks():
    tracks = []
    for layout in ["Brazilian", "Mercosur"]:
        folder = os.path.join(DATASET_ROOT, layout)
        for track_name in sorted(os.listdir(folder)):
            track_path = os.path.join(folder, track_name)
            if os.path.isdir(track_path):
                tracks.append((layout.lower(), track_name, track_path))
    return tracks

def processar_dataset():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "A"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "B"), exist_ok=True)

    tracks = coletar_tracks()
    print(f"Total de tracks encontradas: {len(tracks)}")

    random.seed(SEED)
    random.shuffle(tracks)

    val_count    = int(len(tracks) * VAL_RATIO)
    val_tracks   = tracks[:val_count]
    train_tracks = tracks[val_count:]

    print(f"Treino: {len(train_tracks)} tracks | Validação: {len(val_tracks)} tracks")

    for split_name, split_tracks in [("train", train_tracks), ("val", val_tracks)]:
        print(f"\nProcessando {split_name}...")
        for layout, track_name, track_path in tqdm(split_tracks):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                print(f"JSON não encontrado: {json_path}")
                continue

            with open(json_path, "r") as f:
                ann = json.load(f)
            corners_dict = ann["corners"]

            track_id = track_name.replace("track_", "")

            for idx in range(1, 6):
                frame = f"{idx:03d}"
                hr_file = f"hr-{frame}.png"
                lr_file = f"lr-{frame}.png"

                hr_path = os.path.join(track_path, hr_file)
                lr_path = os.path.join(track_path, lr_file)

                if not os.path.exists(hr_path) or not os.path.exists(lr_path):
                    continue
                if hr_file not in corners_dict or lr_file not in corners_dict:
                    continue

                img_hr = cv2.imread(hr_path)
                img_lr = cv2.imread(lr_path)
                if img_hr is None or img_lr is None:
                    continue

                try:
                    corners_hr = get_corners_from_json(corners_dict, hr_file)
                    corners_lr = get_corners_from_json(corners_dict, lr_file)

                    warped_hr = crop_and_warp(img_hr, corners_hr, TARGET_SIZE_HR)
                    warped_lr = crop_and_warp(img_lr, corners_lr, TARGET_SIZE_LR)

                    nome = f"{layout}_{track_id}_{frame}.png"

                    cv2.imwrite(os.path.join(OUTPUT_ROOT, split_name, "A", nome), warped_hr)  # input
                    cv2.imwrite(os.path.join(OUTPUT_ROOT, split_name, "B", nome), warped_lr)  # alvo

                except Exception as e:
                    print(f"Erro em {track_path}/{frame}: {e}")

    print("\nConcluído!")
    print(f"Dataset salvo em: {OUTPUT_ROOT}")
    print(f"  {OUTPUT_ROOT}/train/A  (HR recortado — input)")
    print(f"  {OUTPUT_ROOT}/train/B  (LR recortado — alvo)")
    print(f"  {OUTPUT_ROOT}/val/A")
    print(f"  {OUTPUT_ROOT}/val/B")

if __name__ == "__main__":
    processar_dataset()
