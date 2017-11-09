gcloud compute --project "woven-goal-180208" ssh --zone "asia-east1-a" "instance-1"

# 
python cifar10_train.py --data_dir ../../../data/train_partial_1

# to run locally
python cifar10_train.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/train_partial \
	--num_class 2 \
	--initial_learn_rate 0.1 


# to run in server
python cifar10_train.py \
	--data_dir ~/data/train_partial \
	--num_class 6 \
	--initial_learn_rate 0.1 \
	--max_steps 10000
