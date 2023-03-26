set -ex
# disable visdom
python train.py --dataroot ~/experiments/data/DEVCOM_5perc \
                --name devcom_nemar \
                --model nemar \
                --gpu_ids 0 \
                --direction BtoA \
              --batch_size 30 \
              --img_height 256 \
              --img_width 256 \
              --display_id 0\
              --dataset_mode aligned