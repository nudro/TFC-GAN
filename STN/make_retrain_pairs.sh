set -ex
# all the image names have to be the same 
# Making a new set of training data that takes the registered images from Epoch 100 of DEVCOM_5Perc training 
# And using these to train for another 50 epochs using hte same algorithm, will be a different model 
# Now have to copy this into /data and add the test set to it (which will be the same original test set)
python combine_A_and_B_mod.py --fold_A ./crop_results/0124_Dev_STN21_200_TrainSet/real_A \
                                --fold_B ./crop_results/0124_Dev_STN21_200_TrainSet/reg_B\
                                --fold_AB ./retrain/0124_Dev_STN21_200_TrainSet