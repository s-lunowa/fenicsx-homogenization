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
ROOTLESS=$(docker info --format '{{ .SecurityOptions }}' | grep "rootless" | wc -l)
NEW_UID=$(id -u)
if [ $ROOTLESS -eq 1 ]; then
  echo "Running in rootless mode. The container will run with the same user privileges as the host user."
  NEW_GID=0
else
  echo "Running in rootful mode. The container will run with root privileges."
  NEW_GID=$(id -g)
fi

# Remove previous containers
echo "Stop and remove any previously created $CONTAINER_NAME container..."
docker stop $CONTAINER_NAME &> /dev/null || true
docker rm $CONTAINER_NAME &> /dev/null || true

# Start new container
echo "Create new container..."

docker run \
  --name ${CONTAINER_NAME} \
  --user root \
  --hostname=${CONTAINER_NAME} \
  --env="GIT_AUTHOR_NAME=$(git config user.name)" \
  --env="GIT_AUTHOR_EMAIL=$(git config user.email)" \
  --env="GIT_COMMIT_NAME=$(git config user.name)" \
  --env="GIT_COMMIT_EMAIL=$(git config user.email)" \
  --volume ${PROJECT_ROOT_DIR}:/home/ubuntu/${CONTAINER_NAME} \
  --workdir /home/ubuntu/${CONTAINER_NAME} \
  -dit ${CONTAINER_NAME} /entrypoint.sh ${NEW_UID} ${NEW_GID}
