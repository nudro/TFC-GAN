set -ex

# registers the entire training dataset for GAN
# uses the best (official) VTF-STN for Devcom: 0302_STN21_Devcom_NewModel3

python make_reg_train_set_refine3_VTFSTN.py --epoch 50 --experiment 0302_STN21_Devcom_NewModel3 --dataset DEVCOM_5perc