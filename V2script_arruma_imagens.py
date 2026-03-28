
import os
from PIL import Image
from tqdm import tqdm

def combine_images(source_root, dest_root, phase):
    dir_A = os.path.join(source_root, phase, 'A')
    dir_B = os.path.join(source_root, phase, 'B')

    dest_dir = os.path.join(dest_root, phase)
    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(dir_A):
        dir_A = os.path.join(source_root, phase, 'A')
        dir_B = os.path.join(source_root, phase, 'B')

    if not os.path.exists(dir_A):
        print(f"ALERTA: Pasta {phase} não encontrada em {source_root}. Pulando...")
        return

    all_images = sorted(os.listdir(dir_A))
    images = all_images[:2000] #podemos limitar aqui
    print(f"Processando {phase}: {len(images)} imagens...")

    for img_name in tqdm(images):
        path_A = os.path.join(dir_A, img_name)
        path_B = os.path.join(dir_B, img_name)

        if os.path.exists(path_B):
            im_A = Image.open(path_A).convert('RGB')
            im_B = Image.open(path_B).convert('RGB')

            if im_A.size != im_B.size:
                im_B = im_B.resize(im_A.size, Image.BICUBIC)

            w, h = im_A.size
            max_dim = max(w, h)

            pad_color = (128, 128, 128)
            sq_A = Image.new('RGB', (max_dim, max_dim), pad_color)
            sq_B = Image.new('RGB', (max_dim, max_dim), pad_color)

            offset_x = (max_dim - w) // 2
            offset_y = (max_dim - h) // 2

            sq_A.paste(im_A, (offset_x, offset_y))
            sq_B.paste(im_B, (offset_x, offset_y))

            combined = Image.new('RGB', (max_dim * 2, max_dim))
            combined.paste(sq_A, (0, 0))             # A na esquerda
            combined.paste(sq_B, (max_dim, 0))       # B na direita

            save_path = os.path.join(dest_dir, img_name)
            combined.save(save_path)

# --- Configuração ---
INPUT_PATH = './sceneA_degradado'
OUTPUT_PATH = './sceneA_aligned'

combine_images(INPUT_PATH, OUTPUT_PATH, 'train')
combine_images(INPUT_PATH, OUTPUT_PATH, 'val')

print("\nConcluído! Dados prontos em:", OUTPUT_PATH)