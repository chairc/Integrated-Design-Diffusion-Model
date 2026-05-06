#!/bin/bash

# Copyright 2025 IDDM Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

IMAGE_NAME="iddm-image"
CONTAINER_NAME="iddm-container"

HOST_DATASET_DIR=""
HOST_OUTPUT_DIR=""
HOST_WEIGHT_DIR=""
HOST_PROJECT_DIR=""

show_help() {
    echo "Usage: bash docker_run.sh [options]"
    echo ""
    echo "Options:"
    echo "  -a, --all       Host project root directory (mapped to /app)"
    echo "  -d, --dataset   Host dataset directory      (mapped to /app/datasets)"
    echo "  -r, --result    Host result directory        (mapped to /app/results)"
    echo "  -w, --weight    Host weight directory        (mapped to /app/weights)"
    echo "  -b, --build     Only build image, do not run container"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Example:"
    echo "  bash docker_run.sh -a /home/user/project"
    echo "  bash docker_run.sh -d /home/user/datasets -r /home/user/results -w /home/user/weights"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--all)
            HOST_PROJECT_DIR="$2"; shift 2 ;;
        -d|--dataset)
            HOST_DATASET_DIR="$2"; shift 2 ;;
        -r|--result)
            HOST_OUTPUT_DIR="$2"; shift 2 ;;
        -w|--weight)
            HOST_WEIGHT_DIR="$2"; shift 2 ;;
        -b|--build)
            BUILD_ONLY=true; shift ;;
        -h|--help)
            show_help; exit 0 ;;
        *)
            echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

echo "=============================="
echo " Step 1: Building Docker image"
echo "=============================="
sudo docker build -t "${IMAGE_NAME}" .
echo "Image '${IMAGE_NAME}' built successfully."
echo ""

if [ "${BUILD_ONLY}" = true ]; then
    echo "Build-only mode. Exiting."
    exit 0
fi

MOUNT_ARGS=""
if [ -n "${HOST_PROJECT_DIR}" ]; then
    if [ -n "${HOST_DATASET_DIR}" ] || [ -n "${HOST_OUTPUT_DIR}" ] || [ -n "${HOST_WEIGHT_DIR}" ]; then
        echo "Warning: -a/--all is specified, individual mounts (-d/-r/-w) will be ignored."
    fi
    mkdir -p "${HOST_PROJECT_DIR}"
    MOUNT_ARGS="${MOUNT_ARGS} -v ${HOST_PROJECT_DIR}:/app"
    echo "Mount project: ${HOST_PROJECT_DIR} -> /app"
else
    if [ -n "${HOST_DATASET_DIR}" ]; then
        mkdir -p "${HOST_DATASET_DIR}"
        MOUNT_ARGS="${MOUNT_ARGS} -v ${HOST_DATASET_DIR}:/app/datasets"
        echo "Mount dataset: ${HOST_DATASET_DIR} -> /app/datasets"
    fi
    if [ -n "${HOST_OUTPUT_DIR}" ]; then
        mkdir -p "${HOST_OUTPUT_DIR}"
        MOUNT_ARGS="${MOUNT_ARGS} -v ${HOST_OUTPUT_DIR}:/app/results"
        echo "Mount results: ${HOST_OUTPUT_DIR} -> /app/results"
    fi
    if [ -n "${HOST_WEIGHT_DIR}" ]; then
        mkdir -p "${HOST_WEIGHT_DIR}"
        MOUNT_ARGS="${MOUNT_ARGS} -v ${HOST_WEIGHT_DIR}:/app/weights"
        echo "Mount weights: ${HOST_WEIGHT_DIR} -> /app/weights"
    fi
fi
echo ""

echo "=============================="
echo " Step 2: Running container"
echo "=============================="
sudo docker run --gpus all -it --rm --name "${CONTAINER_NAME}" \
    ${MOUNT_ARGS} \
    "${IMAGE_NAME}"
