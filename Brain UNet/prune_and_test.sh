#raw run python3 test_script.py --file_write /home/james/brain_unet/brain-segmentation/upsample_conv2d/results_file.txt --CHECKPOINT_PATH /home/james/brain_unet/brain-segmentation/upsample_conv2d/local_prune_0.999999999999999.ckpt

# python3 test_script.py --file_write /home/james/brain_unet/brain-segmentation/bottle_neck/results_file.txt --CHECKPOINT_PATH /home/james/brain_unet/brain-segmentation/bottle_neck/local_prune_0.95.ckpt

echo "Shell Script Starting..."

savefolder="/home/james/brain_unet/brain-segmentation/prune_all"

for ((i=10; i<100; i+=10));
    do
        val=$i
        python3 local_prune.py -amount $val --save_folder $savefolder
    done

python3 local_prune.py -amount 95 --save_folder $savefolder

python3 local_prune.py -amount 99.9999999999 --save_folder $savefolder



temp="/local_prune_0."
checkpoint="${savefolder}${temp}"
temp="/results_file.txt"
write_file="${savefolder}${temp}"

for ((i=1; i<10; i+=1));
    do
        val=$i
        name=".ckpt"
        new="${checkpoint}${val}${name}"
        python3 test_script.py --file_write $write_file --CHECKPOINT_PATH $new
    done

temp="/local_prune_0.95.ckpt"
checkpoint="${savefolder}${temp}"
python3 test_script.py --file_write $write_file --CHECKPOINT_PATH $checkpoint

temp="/local_prune_0.999999999999.ckpt"
checkpoint="${savefolder}${temp}"
python3 test_script.py --file_write $write_file --CHECKPOINT_PATH $checkpoint

echo "... Shell Script Done"
