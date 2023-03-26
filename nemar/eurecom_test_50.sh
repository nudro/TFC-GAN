set -ex
# disable visdom
python test.py --dataroot ~/experiments/data/eurecom_v3_pairs \
                --name eurecom_nemar \
                --model nemar \
                --gpu_ids 0 \
                --direction BtoA \
                --num_test 2000\
                --epoch 50\
              --batch_size 1 \
              --img_height 256 \
              --img_width 256 \
              --dataset_mode aligned