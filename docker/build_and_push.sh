#!/bin/bash

DH_USERNAME="miguelag99"
IMAGE_NAME="tgrip"
TAG_NAME="v1"
USERNAME_IMAGE="perception"
UID=$(id -u)
GID=$(id -g)

set -e

docker build . -t ${IMAGE_NAME}:${TAG_NAME} --build-arg USER=${USERNAME_IMAGE} --build-arg USER_ID=${UID} --build-arg USER_GID=${GID}
docker tag ${IMAGE_NAME}:${TAG_NAME} ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}
docker push ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}
echo "Docker image pushed to DockerHub as ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}"
