help:
	@cat Makefile

DOCKER_NAME=attribute_detection_env

# CUDA version settings
GPU?=0
CUDA_VERSION?=8.0
CUDNN_VERSION?=6

# local folders
PROJ?=$(shell dirname `pwd`)
DATA?=~/wrkdir/datasets

# container folders
CPROJ=/wrkdir/attribute_detection
CDATA=/wrkdir/datasets

DOCKER_FILE=Dockerfile
DOCKER=sudo GPU=$(GPU) nvidia-docker
BACKEND=tensorflow

build:
	sudo docker build -t $(DOCKER_NAME) --build-arg cuda_version=$(CUDA_VERSION) \
        --build-arg cudnn_version=$(CUDNN_VERSION) \
        --build-arg cproj=$(CPROJ) \
        --build-arg cdata=$(CDATA) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(PROJ):$(CPROJ) -v $(DATA):$(CDATA) --env KERAS_BACKEND=$(BACKEND) $(DOCKER_NAME) /bin/bash

ipython: build
	$(DOCKER) run -it -v $(PROJ):$(CPROJ) -v $(DATA):$(CDATA) --env KERAS_BACKEND=$(BACKEND) $(DOCKER_NAME) ipython

notebook: build
	$(DOCKER) run -it -v $(PROJ):$(CPROJ) -v $(DATA):$(CDATA) --net=host --env KERAS_BACKEND=$(BACKEND) $(DOCKER_NAME)

test: build
	$(DOCKER) run -it -v $(PROJ):$(CPROJ) -v $(DATA):$(CDATA) --env KERAS_BACKEND=$(BACKEND) $(DOCKER_NAME) py.test $(CPROJ)

# TODO fix according to project
#app: build
#	$(DOCKER) run -it -v $(PROJ):$(CPROJ) -v $(DATA):$(CDATA) --env KERAS_BACKEND=$(BACKEND) $(DOCKER_NAME) /bin/bash -c "python run_mnist.py -m dev -p ./params/mnist.yaml -n --verbose"
