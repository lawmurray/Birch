FROM opensuse/tumbleweed
RUN zypper --non-interactive --gpg-auto-import-keys addrepo https://download.opensuse.org/repositories/server:database/openSUSE_Tumbleweed/server:database.repo \
    && zypper --non-interactive --gpg-auto-import-keys refresh \
    && zypper --non-interactive --gpg-auto-import-keys install \
    git \
    openssh \
    tar \
    gzip \
    curl \
    ca-certificates \
    moreutils \
    time \
    gcc-c++ \
    make \
    autoconf \
    automake \
    flex \
    bison \
    libtool \
    libexplain-devel \
    libyaml-devel \
    boost-devel \
    libboost_filesystem-devel \
    libboost_system-devel \
    eigen3-devel \
    cairo-devel \
    sqlite3-devel \
    mongo-tools \
    osc
