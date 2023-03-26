while getopts "f": OPTION

# can put into regular evaluation dir since test file set is not necessary to crop, it's just coords

do

  python crop_images.py  --inpath /home/local/AD/cordun1/experiments/TFC-GAN-STN/images/test_results/${OPTARG}\
                          --RA_out /home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/${OPTARG}/real_A\
                          --RB_out /home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/${OPTARG}/real_B \
                          --REGB_out /home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/${OPTARG}/reg_B \
                          --experiment ${OPTARG}

done

