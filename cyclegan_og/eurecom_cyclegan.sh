set -ex

# modified cyclegan script to run nnparallel on 1,2

python cyclegan.py --dataset_name cyclegan_Eurecom\
                    --n_epochs 251\
                    --batch_size 16\
                    --lr 0.001\
                    --b1 0.50\
                    --gpu_num 2\
                    --out_file eurecom_cyclegan\
                    --sample_interval 1000\
                    --checkpoint_interval 50\
                    --experiment eurecom_cyclegan\