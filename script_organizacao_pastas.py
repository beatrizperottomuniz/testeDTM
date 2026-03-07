
#pip install dominate visdom dill scikit-image h5py scipy tqdm opencv-python
#git clone https://github.com/THU-LYJ-Lab/dmt.git

import os
import cv2
import numpy as np
from tqdm import tqdm

DATASET_ROOT = "./tbFcZE-RodoSol-ALPR"
SPLIT_FILE = os.path.join(DATASET_ROOT, "split.txt")

OUTPUT_ROOT = "./rodosol_degradado"

TARGET_SIZE = (128, 64)

def parse_corners(corners_str):
    #string para array
    points = []
    # pegar os pares
    pairs = corners_str.strip().split(' ')
    for pair in pairs:
        x, y = pair.split(',')
        points.append([float(x), float(y)])
    return np.array(points, dtype=np.float32)

def crop_and_warp(img, corners, target_size): #deixa placa reta
    w, h = target_size

    # Pontos de destino (um retângulo perfeito)
    dst_pts = np.array([
        [0, 0],       # Top-Left
        [w-1, 0],     # Top-Right
        [w-1, h-1],   # Bottom-Right
        [0, h-1]      # Bottom-Left
    ], dtype=np.float32)

    # Matriz de transformação
    M = cv2.getPerspectiveTransform(corners, dst_pts)

    # Aplica o warp
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

def degradar_imagem(img_high):
   # degradação bicúbica 
    h, w = img_high.shape[:2]
    # reduz 4x
    small = cv2.resize(img_high, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
    # aumenta de volta
    degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return degraded

def processar_dataset():
    for split in ['train', 'test']:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'A'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'B'), exist_ok=True)

    print(f"Lendo {SPLIT_FILE}...")
    with open(SPLIT_FILE, 'r') as f:
        lines = f.readlines()

    print(f"Processando {len(lines)} imagens...")

    for line in tqdm(lines):
        line = line.strip()
        if not line: continue

        path_part, split_type = line.split(';')

        if path_part.startswith("./"):
            path_part = path_part[2:]

        full_img_path = os.path.join(DATASET_ROOT, path_part)

        if split_type in ['training', 'testing']:
            dest_split = 'train'
        elif split_type == 'validation':
            dest_split = 'test'
        else:
            continue

        img = cv2.imread(full_img_path)
        if img is None:
            print(f"Erro ao abrir imagem: {full_img_path}")
            continue

        #  TXT correspondente para pegar os corners
        txt_path = os.path.splitext(full_img_path)[0] + ".txt"

        if not os.path.exists(txt_path):
            print(f"TXT não encontrado: {txt_path}")
            continue

        # ler corners do TXT
        corners = None
        with open(txt_path, 'r') as f_txt:
            for l_txt in f_txt:
                if l_txt.startswith("corners:"):
                    # Pega tudo depois de "corners: "
                    corners_str = l_txt.split("corners:")[1].strip()
                    corners = parse_corners(corners_str)
                    break

        if corners is None:
            continue

        try:
            img_high = crop_and_warp(img, corners, TARGET_SIZE)

            img_low = degradar_imagem(img_high)

            nome_base = path_part.replace("/", "_").replace("\\", "_")
            if nome_base.startswith("images_"): nome_base = nome_base[7:] 

            cv2.imwrite(os.path.join(OUTPUT_ROOT, dest_split, 'B', nome_base), img_low) #objetivo
            cv2.imwrite(os.path.join(OUTPUT_ROOT, dest_split, 'A', nome_base), img_high) #input

        except Exception as e:
            print(f"Erro ao processar {path_part}: {e}")

    print("\nConcluído!")
    print(f"Dataset salvo em: {OUTPUT_ROOT}")
    print("Estrutura criada:")
    print(f"  {OUTPUT_ROOT}/train/A (input original recortado)")
    print(f"  {OUTPUT_ROOT}/train/B (degradado recortado)")
    print(f"  {OUTPUT_ROOT}/test/A")
    print(f"  {OUTPUT_ROOT}/test/B")

if __name__ == "__main__":
    processar_dataset()