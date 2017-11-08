#!/usr/bin/env bash

IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"


PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=asia-east1
echo $BUCKET_NAME
gsutil mb -l $REGION gs://$BUCKET_NAME
JOB_NAME=job_14_cnn_partial_basic_gpu
JOB_PATH=gs://$BUCKET_NAME/$JOB_NAME
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME/out
IMAGE_PATH=gs://$BUCKET_NAME/data
BOTTLENECK_PATH=gs://$BUCKET_NAME/temp
TEMP_PATH=gs://$BUCKET_NAME/$JOB_NAME/temp
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_PATH \
	--runtime-version 1.2 \
	--module-name trainer.plain_cnn \
	--package-path trainer/ \
	--region $REGION \
	--scale-tier BASIC_GPU \
	-- \
	--verbosity DEBUG \
    --bottleneck_dir $BOTTLENECK_PATH \
    --model_dir $TEMP_PATH \
    --image_dir $IMAGE_PATH \
    --testing_percentage 0 \
    --validation_percentage 20 \
    --train-steps 100


gcloud ml-engine local train \
  --module-name trainer.plain_cnn \
  --package-path trainer/ \
  -- \
  --model_dir '../../../../data/temp/' \
  --bottleneck_dir '../../../../data/temp' \
  --image_dir '../../../../data/train_partial' \
  --testing_percentage 0 \
  --validation_percentage 20 \
  --train-steps 100

python plain_cnn.py \
  --model_dir '../../../../../data/temp/' \
  --bottleneck_dir '../../../../../data/temp' \
  --image_dir '../../../../../data/train_partial' \
  --testing_percentage 0 \
  --validation_percentage 20 \
  --train-steps 500 \
  --learning_rate 0.0001

python plain_cnn.py \
  --model_dir '/tmp/simple_cnn/' \
  --bottleneck_dir '/tmp/simple_cnn/' \
  --image_dir '../../../data/train_partial' \
  --testing_percentage 0 \
  --validation_percentage 20 \
  --train-steps 10000 \
  --learning_rate 0.0001 \  
  --train_batch_size 100



