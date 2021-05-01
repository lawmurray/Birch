FROM ubuntu:latest
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=Europe/London
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ssh \
    tar \
    gzip \
    curl \
    moreutils \
    time \
    ca-certificates \
    binutils \
    elfutils \
    make \
    autoconf \
    automake \
    flex \
    bison \
    g++ \
    gdb \
    libtool \
    libjemalloc-dev \
    libeigen3-dev \
    libyaml-dev \
    libboost-math-dev \
    libsqlite3-dev \
    libcairo2-dev \
    mongo-tools \
    osc \
    doxygen \
    python3-pip \
    && pip3 install mkdocs mkdocs-material \
    && rm -rf /var/lib/apt/lists/*
