Docker container to reproduce Attribute Detection results
=========================================================

This directory contains [docker](https://www.docker.com/) containers to reproduce the system results. To install Docker Community Edition (Docker CE) follow the instructions from [Docker documentation ](https://docs.docker.com/engine/installation/). 

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), some
quick links are here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Running the container

Use ``Makefile`` in order to run the provided Docker container. When container is launched first time, Docker will create the container image by downloading and installing the needed libraries (defined in ``Dockerfile``).

For example, to generate the system results for App run:

    $ make app
    
To open bash shell inside the container environment, run:

    $ make bash

Build the container and start an iPython shell

    $ make ipython

Build the container and start a Jupyter Notebook

    $ make notebook

For GPU support install NVIDIA drivers (ideally latest) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Run using

    $ make notebook GPU=0 # or [ipython, bash]

Switch between Theano and TensorFlow

    $ make notebook BACKEND=theano (old)
    $ make notebook BACKEND=tensorflow

Mount a volume for external data sets

    $ make DATA=~/mydata

Prints all make tasks

    $ make help

You can change Theano parameters by editing `/docker/theanorc`.


Note: If you would have a problem running nvidia-docker you may try the old way
we have used. But it is not recommended. If you find a bug in the nvidia-docker report
it there please and try using the nvidia-docker as described above.

    $ export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run -it -p 8888:8888 $CUDA_SO $DEVICES gcr.io/tensorflow/tensorflow:latest-gpu
