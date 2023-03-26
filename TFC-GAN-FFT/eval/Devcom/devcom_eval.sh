while getopts f:s:m: arg
do
    case $arg in
        f) FILE=$OPTARG
            ;;
        s) SET=$OPTARG
            ;;
        m) MODEL=$OPTARG
            ;;

    esac
done

  # you have to make a 'test_results' dir
  # test_results under Devcom can ONLY include Devcom faces (not like ADAS or Eurecom)
  # manually cp over test_results to devcom/test_results, then use mapping.csv and rename.py to rename to 0 - 1429.png
  # for bhatt and psnr/ssim, test_set defaults to the whole devcom test set. specify `devcom_test_set`
  # m: pass 'custom', this is reserved in case the model is stargan

  python crop_images.py --model ${MODEL} --inpath ${FILE}/test_results --RA_out ${FILE}/real_A --RB_out ${FILE}/real_B --FB_out ${FILE}/fake_B --experiment ${FILE}

  python evaluation_bhatt.py --real_dir ${FILE}/real_B --fake_dir ${FILE}/fake_B --experiment ${FILE} --test_set ${SET}

  python evaluation_psnr_ssim.py --real_dir ${FILE}/real_B --fake_dir ${FILE}/fake_B --experiment ${FILE} --test_set ${SET}
