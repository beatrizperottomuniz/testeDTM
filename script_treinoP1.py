
# !mv /content/rodosol_aligned /content/dmt/TSIT/datasets/

# !ls /content/dmt/TSIT/datasets/

# !cp -r /content/dmt/dmt_utils /content/dmt/TSIT/

#fazer arquivo
# %%writefile /content/dmt/TSIT/data/aligned_dataset.py
# import os.path
# from data.pix2pix_dataset import Pix2pixDataset
# from data.image_folder import make_dataset
# 
# class AlignedDataset(Pix2pixDataset):
#     """
#     Dataset simples que herda de Pix2pixDataset.
#     Serve apenas para dizer onde estão os arquivos (get_paths).
#     """
# 
#     def get_paths(self, opt):
#         # O código pega o caminho que você passou no --croot
#         root = opt.croot  # ./datasets/rodosol_aligned
#         phase = opt.phase # 'train'
# 
#         # Monta o caminho: ./datasets/rodosol_aligned/train
#         dir_path = os.path.join(root, phase)
# 
#         # Busca todas as imagens dentro dessa pasta
#         all_images = sorted(make_dataset(dir_path))
# 
#         # O código original pede 3 retornos: (Labels, Imagens Reais, Instancias)
#         # Como suas imagens já têm A e B juntas, retornamos a mesma lista para tudo.
#         return all_images, all_images, all_images
# 
#     def name(self):
#         return 'AlignedDataset'


# Arquivo que está dando erro
arquivo_modelo = 'dmt/TSIT/models/pix2pix_model.py'

with open(arquivo_modelo, 'r') as f:
    codigo = f.read()

codigo_novo = codigo.replace('betas=(beta1, beta2)', 'betas=(float(beta1), float(beta2))')

with open(arquivo_modelo, 'w') as f:
    f.write(codigo_novo)

print("✅ Correção aplicada com sucesso no pix2pix_model.py!")

