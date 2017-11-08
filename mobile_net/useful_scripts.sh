IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

# original 

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos

# ---------------------------------------
tensorboard --logdir tf_files/training_summaries &


python -m scripts.label_image \
    --mass_label=true  \
    --input_folder=tf_files/test_photos
    # --input_folder=../../../../data/transferred_test


python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=../../../../data/transferred_test/0.jpg

python -m scripts.label_image2 \
    --mass_label=true  \
    --input_folder=tf_files/test_photos \
    --save_to_file=result1.csv

python -m scripts.label_image2 \
    --mass_label=true  \
    --input_folder=tf_files/test_photos2 \
    --save_to_file=result2.csv