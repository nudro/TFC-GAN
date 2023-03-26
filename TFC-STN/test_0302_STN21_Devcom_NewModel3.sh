set -ex
# Training on the original Devcom dataset, but with this new kind of model 
python test_TFCGAN_STN21_refine3.py --dataset DEVCOM_5perc --experiment 0302_STN21_Devcom_NewModel3 --epoch 50