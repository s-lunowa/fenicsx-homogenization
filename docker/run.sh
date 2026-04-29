#!/bin/bash
#
# Run the docker image
#
# Author: S.B. Lunowa

# Exit on failure
set -e

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT_DIR="${DIR}/.."
CONTAINER_NAME="fenicsx-homogenization"

# Remove previous containers
echo "Stop and remove any previously created $CONTAINER_NAME container..."
docker stop $CONTAINER_NAME &> /dev/null || true
docker rm $CONTAINER_NAME &> /dev/null || true

# Start new container
echo "Create new container..."

docker run \
  --name $CONTAINER_NAME \
  --user root \
  --net=host \
  --pid=host \
  --ipc=host \
  --hostname=docker-container \
  --privileged \
  --security-opt seccomp=unconfined \
  --env="GIT_AUTHOR_NAME=$(git config user.name)" \
  --env="GIT_AUTHOR_EMAIL=$(git config user.email)" \
  --env="GIT_COMMIT_NAME=$(git config user.name)" \
  --env="GIT_COMMIT_EMAIL=$(git config user.email)" \
  --volume ${PROJECT_ROOT_DIR}:${PROJECT_ROOT_DIR} \
  --volume /dev/shm:/dev/shm \
  --volume /lib/modules:/lib/modules \
  --volume /dev/dri:/dev/dri \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --workdir $(pwd) \
  -dit $CONTAINER_NAME /entrypoint.sh `id -u` `id -g`
