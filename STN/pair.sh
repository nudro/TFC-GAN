while getopts "f": OPTION

do

python combine_A_and_B_mod.py \
    --experiment ${OPTARG}\
    --fold_A /home/local/AD/cordun1/experiments/TFC-GAN-STN/${OPTARG}/real_A \
    --fold_B /home/local/AD/cordun1/experiments/TFC-GAN-STN/${OPTARG}/real_B \
    --fold_AB /home/local/AD/cordun1/experiments/TFC-GAN-STN/${OPTARG}/pairs/real/ \



done
