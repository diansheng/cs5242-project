gcloud compute --project "woven-goal-180208" ssh --zone "asia-east1-a" "instance-1"

# 
python cifar10_train.py --data_dir ../../../data/train_partial_1

# to run locally
python cifar10_train.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/train_partial \
	--num_class 2 \
	--initial_learn_rate 0.1 


# to run locally without training data restructure
python cifar10_train.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/transferred_train \
	--num_class 2 \
	--initial_learn_rate 0.1 

# to run in server
python cifar10_train.py \
	--data_dir ~/data/train_partial \
	--num_class 6 \
	--initial_learn_rate 0.1 \
	--batch_size 30 \
	--max_steps 10000


# evaluate
python cifar10_eval.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/transferred_train_partial \
	--eval_data train \
	--num_examples 200 \


python cifar10_train.py --data_dir ~/data/train_full --num_class 132 --initial_learn_rate 0.1 --batch_size 50 


"""
loss improve log
2 -> 1.1, batch_size 100 -> 30
1.1 -> 0.4, num_epoch_learn_rate_decay 350 -> 1000
0.4 -> 0.07, num_epoch_learn_rate_decay 1000 -> 3000
"""
