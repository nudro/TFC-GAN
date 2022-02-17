set -ex
# MULTIGPU on 1 and 2
python favtgan_Global_Temp_MULTIGPU_basemodel_AMP_blurpool_TripTemp_16Patch.py --dataset_name eurecom_v3_pairs\
                                                                                --n_epochs 1001 \
                                                                                --batch_size 64 \
                                                                                --gpu_num 1 \
                                                                                --out_file 0203_eurecom_TripTemp_16 \
                                                                                --sample_interval 50 \
                                                                                --checkpoint_interval 100 \
                                                                                --experiment 0203_eurecom_TripTemp_16\
                                                                                --lr 0.001\
                                                                                --b1 0.50\
