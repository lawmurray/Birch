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
    make \
    autoconf \
    automake \
    flex \
    bison \
    g++ \
    gdb \
    libtool \
    libeigen3-dev \
    libyaml-dev \
    libboost-math-dev \
    libsqlite3-dev \
    libcairo2-dev \
    mongo-tools \
    osc \
    && rm -rf /var/lib/apt/lists/*
