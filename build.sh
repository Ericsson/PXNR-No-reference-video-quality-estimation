#!/usr/bin/env bash
#
# Build the PXNR Docker image.
#
#   ./build.sh                    Build for THIS machine's architecture and load
#                                 it into local Docker. Run it with:
#                                     docker run --rm pxnr ...
#
#   ./build.sh gpu                Build the NVIDIA/CUDA image (linux/amd64 only)
#                                 and load it locally as pxnr:gpu. Run it with:
#                                     docker run --rm --gpus all pxnr:gpu ...
#                                 Requires the NVIDIA Container Toolkit on the
#                                 host (see README, "Docker GPU").
#
#   ./build.sh push <image-ref>   Build a multi-arch image (linux/amd64 +
#                                 linux/arm64) and push it to a registry, e.g.
#                                     ./build.sh push ghcr.io/aveq/pxnr:latest
#                                 (A multi-arch image can't be loaded into the
#                                 local store, so it must be pushed.)
#
set -euo pipefail

IMAGE="pxnr:latest"
DOCKERFILE="Dockerfile"
BUILDER="pxnr-multiarch"

cd "$(dirname "$0")"

# A docker-container builder is required for multi-platform builds; create
# it once if it isn't there yet.
if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
    echo ">> creating buildx builder '$BUILDER'"
    docker buildx create --name "$BUILDER" --driver docker-container --bootstrap >/dev/null
fi

if [[ "${1:-}" == "gpu" ]]; then
    echo ">> building linux/amd64 GPU image and loading into local Docker as pxnr:gpu"
    docker buildx build --builder "$BUILDER" \
        --platform linux/amd64 \
        -f "Dockerfile.gpu" -t "pxnr:gpu" --load .
elif [[ "${1:-}" == "push" ]]; then
    REF="${2:?usage: ./build.sh push <image-ref>}"
    echo ">> building multi-arch (linux/amd64 + linux/arm64) and pushing to $REF"
    docker buildx build --builder "$BUILDER" \
        --platform linux/amd64,linux/arm64 \
        -f "$DOCKERFILE" -t "$REF" --push .
else
    case "$(uname -m)" in
        x86_64|amd64)  PLATFORM=linux/amd64 ;;
        aarch64|arm64) PLATFORM=linux/arm64 ;;
        *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    echo ">> building $PLATFORM and loading into local Docker as $IMAGE"
    docker buildx build --builder "$BUILDER" \
        --platform "$PLATFORM" \
        -f "$DOCKERFILE" -t "$IMAGE" --load .
fi

echo ">> done"
