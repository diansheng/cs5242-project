# for file in ../../../../data/*/ ; do 
#   if [[ -d "$file" && ! -L "$file" ]]; then
#     echo "$file is a directory"; 
#   fi; 
# done

for file in $(find /Users/diansheng/Codebox/tensorflow_env/cs5242/data/test/ -mindepth 1 -maxdepth 1 -type d); do 
# for file in $(find tf_files/test_photos/ -mindepth 1 -maxdepth 1 -type d); do 
    dir=${file%*/}
    dir=${dir##*/}
    echo $dir
    python -m scripts.label_image2 \
	    --mass_label=true  \
	    --input_folder=../../../../data/test/$dir \
	    --save_to_file=$dir.csv
done