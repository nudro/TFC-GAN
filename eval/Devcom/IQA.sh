while getopts e:d: arg 
do
    case $arg in
        e) EXP=$OPTARG 
            ;;
        d) DATA=$OPTARG
            ;;

    esac
done

   
    python IQA-PyTorch/inference_iqa.py \
                            -m maniqa \
                            -i /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/$EXP/fake_B \
                            -r /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/$EXP/real_B \
                            --save_file /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/metrics/{$EXP}_MANIQA.txt
                            
    python IQA-PyTorch/inference_iqa.py \
                            -m dbcnn\
                            -i /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/$EXP/fake_B \
                            -r /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/$EXP/real_B\
                            --save_file /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/metrics/{$EXP}_DBCNN.txt
   
    python IQA-PyTorch/inference_iqa.py \
                            -m niqe\
                            -i /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/$EXP/fake_B\
                            -r /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/$EXP/real_B\
                            --save_file /home/local/AD/cordun1/experiments/TFC-GAN/evaluation/$DATA/metrics/{$EXP}_NIQE.txt