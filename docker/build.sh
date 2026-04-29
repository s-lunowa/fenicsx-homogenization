#!/bin/bash
#
# Build the docker image
# run `docker/build.sh` to build the image.
#
# Author: S.B. Lunowa

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CONTAINER_NAME="fenicsx-homogenization"

echo "Build docker image '$name' with : "
docker --version

if [ "$(uname)" == "Darwin" ]; then
    docker buildx build $DIR --network=host --platform linux/arm64 \
           -t $CONTAINER_NAME --load
else
    docker build $DIR --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
           --network=host -t $CONTAINER_NAME
fi
