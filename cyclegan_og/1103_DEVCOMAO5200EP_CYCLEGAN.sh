set -ex
python cyclegan.py --dataset_name DEVCOM_AO_Pairs_5Perc_CycleGAN_Style\
                    --n_epochs 201\
                    --batch_size 32\
                    --gpu_num 2\
                    --out_file 1103_DEVCOMAO5200EP_CYCLEGAN\
                    --sample_interval 1000\
                    --checkpoint_interval 50\
                    --experiment 1103_DEVCOMAO5200EP_CYCLEGAN\