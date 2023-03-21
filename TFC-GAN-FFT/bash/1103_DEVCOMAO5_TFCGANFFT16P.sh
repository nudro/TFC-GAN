# set to GPUs 0,2
set -ex

python TFCGAN_multigpu_patchFFT_16P.py --dataset_name DEVCOM_AO_Pairs_5Perc\
                                    --n_epochs 201 \
                                    --batch_size 32 \
                                    --gpu_num 0 \
                                    --out_file 1103_DEVCOMAO5_TFCGANFFT16P\
                                    --sample_interval 200 \
                                    --checkpoint_interval 100 \
                                    --experiment 1103_DEVCOMAO5_TFCGANFFT16P \
