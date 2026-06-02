# Dockerfile for FEniCSx Homogenization examples
# Based on Ubuntu 24.04 with FEniCSx installed from the fenics-packages PPA
#
# Author: S.B. Lunowa

FROM ubuntu:24.04

# This will avoid tzdata package from requesting user interaction
RUN apt update && apt install -y \
  bash-completion \
  man \
  nano \
  software-properties-common \
  time \
  python3-pip
RUN if ! [ -L /etc/localtime ]; then \
    ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime; \
  fi

# Install further packages and clear cache
RUN add-apt-repository ppa:fenics-packages/fenics -y
RUN apt update && apt install -y \
  git \
  fenicsx \
  gmsh \
  python3-gmsh \
  python3-setuptools \
  && apt clean

# Setup a config for Git
RUN git config --system user.name "test" && \
  git config --system user.email test@test.com

# Handle the user
USER ubuntu

# copy requirements
COPY requirements.txt /tmp/requirements.txt

# install python deps
RUN python3 -m pip install --break-system-packages --upgrade pip && \
  python3 -m pip install --break-system-packages -r /tmp/requirements.txt

# Recursive chown is incredibly slow in docker https://github.com/docker/for-linux/issues/388
# -> Give other users full permission to skip this step when UID and GID are changed.
RUN chmod o=rwX -R $HOME

# Create entrypoint
COPY entrypoint.sh /entrypoint.sh 
ENTRYPOINT ["/bin/bash"]
