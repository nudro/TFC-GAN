set -ex
# Training on the original Devcom dataset, but with this new kind of model 
python TFCGAN_STN21_Original_refine3.py --dataset DEVCOM_5perc --experiment 0302_STN21_Devcom_NewModel3 --batch_size 32 --gpu_num 1 --n_epochs 100