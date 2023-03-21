# 3 GPUs
set -ex
python TFCGAN_multigpu_globalFFT_16P.py --dataset DEVCOM_AO_Pairs_5Perc \
                            --experiment 1104_DEVCOMAO5_TFCGANFFT16P_Global\
                            --batch_size 32 \
                            --gpu_num 0 \
                            --n_epochs 201\
                            --sample_interval 200 \
                            --checkpoint_interval 50 \
                            --out_file 1104_DEVCOMAO5_TFCGANFFT16P_Global\