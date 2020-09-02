FROM opensuse/tumbleweed
RUN zypper --non-interactive addrepo --no-gpgcheck "https://repo.mongodb.org/zypper/suse/15/mongodb-org/4.4/x86_64/" mongodb \
    && zypper --non-interactive refresh \
    && zypper --non-interactive install --no-recommends \
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
    mongodb-database-tools \
    osc \
    && zypper clean --all
