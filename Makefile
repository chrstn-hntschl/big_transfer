build:
	docker build -t google/big_transfer .

models/BiT-M-R50x1.h5:
	wget -O models/BiT-M-R50x1.h5 https://storage.googleapis.com/bit_models/BiT-M-R50x1.h5
	
CIFAR10_5examples_per_class:
	docker run --rm -it --gpus all --name cifar10_`date +%F_%H%M%S` --bit_pretrained_dir /opt/big_transfer_models --model BiT-M-R50x1 --batch 128 --base_lr 0.001  --logdir /tmp/bit_logs --dataset cifar10 --examples_per_class 5 --examples_per_class_seed 0
