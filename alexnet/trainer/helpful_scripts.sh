gcloud compute --project "woven-goal-180208" ssh --zone "asia-east1-a" "instance-1"

# 
python cifar10_train.py --data_dir ../../../data/train_partial_1

# to run locally
python cifar10_train.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/transferred_train \
	--num_class 2 \
	--initial_learn_rate 0.1 \
	--batch_size 32 \
	--max_steps 1000

# to run locally without training data restructure
python cifar10_train.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/transferred_train \
	--num_class 132 \
	--initial_learn_rate 0.1 \
	--batch_size 30 \
	--max_steps 100

python cifar10_multi_gpu_train.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/transferred_train \
	--num_class 132 \
	--initial_learn_rate 0.1 \
	--batch_size 32 \
	--num_gpus 2

# evaluate
python cifar10_eval.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/transferred_train \
	--eval_data train_eval \
	--num_examples 557 \
	--eval_interval_secs 10

# predict
python cifar10_eval.py \
	--data_dir ~/Codebox/tensorflow_env/cs5242/data/transferred_test \
	--eval_data test \
	--num_examples 557 \
	--eval_interval_secs 10 \
	--run_once true

####################################################

# to run in server
# full class in gcloud
python cifar10_train.py --data_dir ~/data/transferred_train --num_class 132 --initial_learn_rate 0.1 --batch_size 50 

nohup python cifar10_train.py \
	--data_dir ~/data/transferred_train \
	--num_class 132 \
	--initial_learn_rate 0.1 \
	--batch_size 50 \
	--max_steps 10000 \
	> nohup.log &


nohup python cifar10_multi_gpu_train.py \
	--data_dir ~/data/transferred_train \
	--num_class 132 \
	--initial_learn_rate 0.5 \
	--batch_size 50 \
	--max_steps 100000 \
	--num_gpus 2 > nohup.log &

python cifar10_eval.py \
	--data_dir ~/data/transferred_train \
	--eval_data train_eval \
	--num_examples 1000 \
	--eval_interval_secs 10 \
	--num_class 132 \
	--run_once true

python cifar10_eval.py \
	--data_dir ~/data/transferred_train \
	--eval_data train_eval \
	--num_examples 1000 \
	--eval_interval_secs 10 \
	--num_class 66 \
	--run_once true


nohup python cifar10_train.py --data_dir ~/data/transferred_train --num_class 132 --initial_learn_rate 0.1 --batch_size 50 > nohup.log &

"""
loss improve log
2 -> 1.1, batch_size 100 -> 30
1.1 -> 0.4, num_epoch_learn_rate_decay 350 -> 1000
0.4 -> 0.07, num_epoch_learn_rate_decay 1000 -> 3000

original: 3.7



"""
