set -ex
python cyclegan.py --dataset_name cyclegan_Eurecom\
                    --n_epochs 201\
                    --batch_size 32\
                    --gpu_num 0\
                    --out_file 1105_EUR200EP_CYCLEGAN\
                    --sample_interval 1000\
                    --checkpoint_interval 50\
                    --experiment 1105_EUR5200EP_CYCLEGAN\