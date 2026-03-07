
import sys
import os

# pasta certa
# cd ./dmt/TSIT

# 2. Adiciona a pasta atual ao caminho do Python (para ele achar os módulos)
sys.path.append(os.getcwd())

# 3. Verifica se a pasta sync_batchnorm existe (Diagnóstico)
#path_check = "models/networks/sync_batchnorm"
#if not os.path.exists(path_check):
#    print(f"ALERTA: A pasta {path_check} não existe! Baixando agora...")
#    # Baixa o código que falta (Synchronized-BatchNorm-PyTorch)
#    git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch models/networks/sync_batchnorm
#    !cp models/networks/sync_batchnorm/sync_batchnorm/* models/networks/sync_batchnorm/
#else:
#    print(f"A pasta {path_check} existe.")


# roda o treino
# !python train.py \
#   --croot ./datasets/rodosol_aligned \
#   --name rodosol_v1 \
#   --model dmt \
#   --dataset_mode aligned \
#   --batchSize 4 \
#   --gpu_ids 0 \
#   --no_flip \
#   --preprocess_mode scale_width_and_crop \
#   --load_size 256 \
#   --crop_size 256 \
#   --aspect_ratio 1.0
