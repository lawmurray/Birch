FROM ubuntu:20.04
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=Europe/London
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ssh \
    tar \
    gzip \
    curl \
    ca-certificates \
    g++ \
    make \
    autoconf \
    automake \
    flex \
    bison \
    libtool \
    libexplain-dev \
    libeigen3-dev \
    libyaml-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-math-dev \
    libsqlite3-dev \
    libcairo2-dev \
    mongo-tools \
    && rm -rf /var/lib/apt/lists/*
