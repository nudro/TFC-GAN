while getopts f: arg
do
    case $arg in
        f) FILE=$OPTARG
            ;;

    esac
done

  python Eurecom_MagMSE.py --real_dir ${FILE}/real_B --fake_dir ${FILE}/fake_B --experiment ${FILE} --test_set eurecom_test_set