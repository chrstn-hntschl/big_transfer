IMAGE_NAME=google/big_transfer
IMAGE_VER=latest

PRETRAINED_MODEL?=BiT-M-R50x1


cur_dir = $(shell pwd)

build:
	echo "models/*" >> .dockerignore
	echo "datasets/*" >> .dockerignore
	docker build -t ${IMAGE_NAME}:${IMAGE_VER} .
	rm .dockerignore

models/${PRETRAINED_MODEL}.h5:
	wget -O models/${PRETRAINED_MODEL}.h5 https://storage.googleapis.com/bit_models/${PRETRAINED_MODEL}.h5
	
CIFAR10_5examples_per_class: build models/${PRETRAINED_MODEL}.h5
	docker run --rm -it --gpus all -v $(cur_dir)/models:/opt/big_transfer_models:ro -v $(cur_dir)/datasets:/root/tensorflow_datasets ${IMAGE_NAME}:${IMAGE_VER} --name cifar10_`date +%F_%H%M%S` --bit_pretrained_dir /opt/big_transfer_models --model ${PRETRAINED_MODEL} --batch 128 --base_lr 0.001  --logdir /tmp/bit_logs --dataset cifar10 --examples_per_class 5 --examples_per_class_seed 0

# BiT-M-R50x1, mAP=0.4411, BiT-M-R101x3, mAP=
WIKIPAINTINGS5_all: build models/${PRETRAINED_MODEL}.h5
	docker run --rm -it --gpus all -v $(cur_dir)/models:/opt/big_transfer_models:ro -v $(cur_dir)/datasets:/root/tensorflow_datasets ${IMAGE_NAME}:${IMAGE_VER} --name wikipaintings_5_`date +%F_%H%M%S` --bit_pretrained_dir /opt/big_transfer_models --model ${PRETRAINED_MODEL} --batch 64 --base_lr 0.001  --logdir /tmp/bit_logs --dataset wikipaintings --dataset_config Wikipaintings_5

dev: build
	docker run --rm -it --gpus all -v $(cur_dir):/opt/big_transfer -v $(cur_dir)/datasets:/root/tensorflow_datasets --entrypoint /bin/bash ${IMAGE_NAME}:${IMAGE_VER}
