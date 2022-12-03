FROM opensuse/leap:latest
RUN zypper --non-interactive addrepo --no-gpgcheck "https://repo.mongodb.org/zypper/suse/15/mongodb-org/4.4/x86_64/" mongodb \
    && zypper --non-interactive refresh \
    && zypper --non-interactive update \
    && zypper --non-interactive install --no-recommends \
    git \
    openssh \
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
    gcc-c++ \
    gdb \
    libtool \
    libyaml-devel \
    boost-devel \
    eigen3-devel \
    cairo-devel \
    sqlite3-devel \
    mongodb-database-tools \
    osc \
    gh \
    doxygen \
    python310 \
    python310-pip \
    && pip3-3.10 install mkdocs mkdocs-material \
    && zypper clean --all

