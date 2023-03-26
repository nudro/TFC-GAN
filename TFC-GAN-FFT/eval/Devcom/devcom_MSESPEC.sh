while getopts f: arg
do
    case $arg in
        f) FILE=$OPTARG
            ;;

    esac
done

  python Devcom_MagMSE.py --real_dir ${FILE}/real_B --fake_dir ${FILE}/fake_B --experiment ${FILE} --test_set devcom_5perc_test_set