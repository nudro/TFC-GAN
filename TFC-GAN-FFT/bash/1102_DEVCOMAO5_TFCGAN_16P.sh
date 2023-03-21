# 3 GPUs
set -ex
python TFCGAN_original_16P.py --dataset DEVCOM_AO_Pairs_5Perc \
                            --experiment 1102_DEVCOMAO5_TFCGAN_16P\
                            --batch_size 32 \
                            --gpu_num 0 \
                            --n_epochs 201\
                            --sample_interval 200 \
                            --checkpoint_interval 50 \