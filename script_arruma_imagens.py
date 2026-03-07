
import os
from PIL import Image
from tqdm import tqdm

def combine_images(source_root, dest_root, phase):
    # Caminhos de entrada (ajuste se suas pastas forem maiúsculas A/B)
    dir_A = os.path.join(source_root, phase, 'a')
    dir_B = os.path.join(source_root, phase, 'b')

    # Caminho de saída
    dest_dir = os.path.join(dest_root, phase)
    os.makedirs(dest_dir, exist_ok=True)

    # Verifica se as pastas existem (tenta maiúsculo se falhar)
    if not os.path.exists(dir_A):
        dir_A = os.path.join(source_root, phase, 'A')
        dir_B = os.path.join(source_root, phase, 'B')

    if not os.path.exists(dir_A):
        print(f"ALERTA: Pasta {phase} não encontrada em {source_root}. Pulando...")
        return

    # Lista imagens
    images = sorted(os.listdir(dir_A))
    print(f"Processando {phase}: {len(images)} imagens...")

    for img_name in tqdm(images):
        path_A = os.path.join(dir_A, img_name)
        path_B = os.path.join(dir_B, img_name)

        if os.path.exists(path_B):
            im_A = Image.open(path_A).convert('RGB')
            im_B = Image.open(path_B).convert('RGB')

            # Redimensiona B para caber em A se necessário
            if im_A.size != im_B.size:
                im_B = im_B.resize(im_A.size, Image.BICUBIC)

            # Cria a imagem combinada (Dobro da largura)
            w, h = im_A.size
            combined = Image.new('RGB', (w * 2, h))
            combined.paste(im_A, (0, 0))    # A na esquerda
            combined.paste(im_B, (w, 0))    # B na direita

            # Salva na pasta nova
            save_path = os.path.join(dest_dir, img_name)
            combined.save(save_path)

# --- Configuração ---
# Onde estão seus dados agora (separados)
INPUT_PATH = './rodosol_degradado'
# Onde vamos salvar os dados prontos (juntos)
OUTPUT_PATH = './rodosol_aligned'

combine_images(INPUT_PATH, OUTPUT_PATH, 'train')
combine_images(INPUT_PATH, OUTPUT_PATH, 'test')

print("\nConcluído! Dados prontos em:", OUTPUT_PATH)
