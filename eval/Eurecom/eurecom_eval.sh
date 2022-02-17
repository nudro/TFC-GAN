while getopts f:m: arg
do
    case $arg in
        f) FILE=$OPTARG
            ;;
        m) MODEL=$OPTARG
            ;;

    esac
done

  # test_results under Eurecom can ONLY include Eurecom faces (not like ADAS or Devcom)
  # manually cp over test_results to eurecom/test_results
  python crop_images.py --model ${MODEL} --inpath ${FILE}/test_results --RA_out ${FILE}/real_A --RB_out ${FILE}/real_B --FB_out ${FILE}/fake_B --experiment ${FILE}

  python evaluation_bhatt.py --real_dir ${FILE}/real_B --fake_dir ${FILE}/fake_B --experiment ${FILE}

  python evaluation_psnr_ssim.py --real_dir ${FILE}/real_B --fake_dir ${FILE}/fake_B --experiment ${FILE}
