set -ex

# full DEVCOM dataset

python cyclegan.py --dataset_name DEVCOM_AO_CycleGAN_Style\
                    --n_epochs 201\
                    --batch_size 32\
                    --gpu_num 0\
                    --out_file 1212_DEVCOMAOFULL_CYCLEGAN\
                    --sample_interval 1000\
                    --checkpoint_interval 50\
                    --experiment 1212_DEVCOMAOFULL_CYCLEGAN\