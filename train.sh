#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
#torchrun --nproc_per_node=5 train.py --stage 1
#
### torchrun --nproc_per_node=10 train.py --stage 2
#
### torchrun --nproc_per_node=10 train.py --stage 3

#!/bin/bash
# ==============================
# 单机多卡分布式训练启动脚本
# ==============================

# 你想使用的 GPU 数量
NUM_GPUS=4

# 数据路径与保存路径（根据你情况修改）
DATA_PATH="/home/liuwei/mnt/instant_vggt_dataset/mask_train"
SAVE_DIR="./ckpts"

# 选择是否使用混合精度（True/False）
AMP=True

# 每多少个epoch保存一次
SAVE_INTERVAL=1

# 预训练模型路径（如果没有就留空）
PRETRAINED="./ckpts/pi3_sky_best.safetensors"

# 启动训练
CUDA_VISIBLE_DEVICES=0 \
python3 -u train_sky.py \
    --gpus ${NUM_GPUS} \
    --data-path ${DATA_PATH} \
    --save-dir ${SAVE_DIR} \
    --amp ${AMP} \
    --save-interval ${SAVE_INTERVAL} \
    --pretrained-vggt ${PRETRAINED} \
    --sky-epochs 60 \
    --train-sky True
