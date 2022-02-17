set -ex
# now favtgan_Global_Temp_MULTIGPU_basemodel_AMP_blurpool_L1
python favtgan_Global_Temp_MULTIGPU_basemodel_AMP_blurpool.py --dataset_name eurecom_v3_pairs\
                                                                --n_epochs 1001 \
                                                                --batch_size 64 \
                                                                --gpu_num 0 \
                                                                --out_file 0124_eurecom_temp \
                                                                --sample_interval 50 \
                                                                --checkpoint_interval 100 \
                                                                --experiment 0124_eurecom_temp\
                                                                --lr 0.001\
                                                                --b1 0.50\
