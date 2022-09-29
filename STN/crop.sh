while getopts "f": OPTION

# /home/local/AD/cordun1/experiments/TFC-GAN/images/test_results/0901_STN_V8_OG_fBA

do

  python crop_stn_stack.py --inpath /home/local/AD/cordun1/experiments/TFC-GAN/images/test_results/${OPTARG} \
                              --RA_out /home/local/AD/cordun1/experiments/TFC-GAN-STN/${OPTARG}/real_A \
                                --RB_out /home/local/AD/cordun1/experiments/TFC-GAN-STN/${OPTARG}/real_B \
                                --RegB_out /home/local/AD/cordun1/experiments/TFC-GAN-STN/${OPTARG}/reg_B \
                                --experiment ${OPTARG}
done
