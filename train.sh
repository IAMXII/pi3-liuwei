CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
torchrun --nproc_per_node=5 train.py --stage 1

## torchrun --nproc_per_node=10 train.py --stage 2

## torchrun --nproc_per_node=10 train.py --stage 3