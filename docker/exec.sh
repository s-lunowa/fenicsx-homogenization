#!/bin/bash
#
# Start interactive session in the docker container
#
# Author: S.B. Lunowa

CONTAINER_NAME="fenicsx-homogenization"
docker exec --user ubuntu -it $CONTAINER_NAME /bin/bash
