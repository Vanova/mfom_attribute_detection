# Environment: Conda, Python(2.7/3.5), Tensorflow(1.4.1), Keras(2.0.2)

ARG cuda_version=8.0
ARG cudnn_version=6
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel-ubuntu14.04

# Install extra packages if required
RUN apt-get update -qq && apt-get install --no-install-recommends -y \
    # Essentials
    build-essential \
    g++ \
    git \
    locales \
    curl \
    libsndfile1 \
    sndfile-tools \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get clean
ENV LC_ALL C.UTF-8

###
# Take appropriate CONDA settings
###

# use python2
COPY conda/ai.py2.yml /wrkdir/env/ai.yml

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda-latest-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:$PATH > /etc/profile.d/conda.sh
RUN conda update -y conda && \
    pip install --upgrade pip

# Add the user that will run the app (no need to run as root)
# Working directory for data and projects
ENV NB_USER vano
ENV NB_UID 1000

ARG cproj=/wrkdir/projects
ARG cdata=/wrkdir/datasets

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p $cproj && \
    mkdir -p $cdata && \
    chown $NB_USER /wrkdir
USER $NB_USER

RUN conda env create -f /wrkdir/env/ai.yml && \
    rm -rf /opt/conda/pkgs/* && \
    conda clean -yt

# Install project into container
WORKDIR ${cproj}

# Activate the environment
ENV PATH $CONDA_DIR/envs/ai/bin:$PATH
ENV PATH $HOME/bin:$PATH
ENV PYTHONPATH ${cproj}:$PYTHONPATH