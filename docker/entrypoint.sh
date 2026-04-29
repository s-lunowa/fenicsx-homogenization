#!/bin/bash
#
# Update user and group id of the precompiled docker image
# to match the host system.
#
# Author: S.B. Lunowa

if [[ $# != 2 ]]; then
  echo "Wrong number of arguments.
Description:
  Update user id and group id of the precompiled image such that 
  mapped volumes can be accessed with the same rights as on the host. 
  Files created by the container can also be accessed on the host without chowning.
Usage:
  $0 NEW_UID NEW_GID
  "
  exit 1
fi

set -e
NEW_UID=$1
shift
NEW_GID=$1

echo "127.0.1.1   `hostname`" >> /etc/hosts

# using usermod hangs for a long time:
# Bug report https://github.com/containers/podman/issues/1808
# -> Use sed
sed -i "s|ubuntu:x:1000:|ubuntu:x:${NEW_GID}:|g" /etc/group
sed -i "s|ubuntu:x:1000:1000:Ubuntu|ubuntu:x:${NEW_UID}:${NEW_GID}:Ubuntu|g" /etc/passwd
# chowning not necessary because in the Dockerfile we gave full permission to other users.
# Reason: recursive chown is incredibly slow in docker https://github.com/docker/for-linux/issues/388
/bin/bash
