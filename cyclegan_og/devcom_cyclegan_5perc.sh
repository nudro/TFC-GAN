set -ex
python cyclegan.py --dataset_name cyclegan_Devcom_5perc\
                    --n_epochs 251\
                    --batch_size 16\
                    --lr 0.001\
                    --b1 0.50\
                    --gpu_num 2\
                    --out_file devcom_cyclegan_5perc\
                    --sample_interval 1000\
                    --checkpoint_interval 50\
                    --experiment devcom_cyclegan_5perc\