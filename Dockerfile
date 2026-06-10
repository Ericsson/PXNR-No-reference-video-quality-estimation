# syntax=docker/dockerfile:1
#
# Multi-platform build for PXNR (linux/amd64 + linux/arm64).
#
# The official tensorflow/tensorflow image is amd64-only, so it can't be
# used for a native arm64 build. Instead we start from the multi-arch
# python image and install TensorFlow from PyPI (see requirements.txt).
#
# Build with the helper script:  ./build.sh           (native, loadable)
#                                ./build.sh push REF  (amd64+arm64, pushed)
FROM python:3.12-slim

# TARGETARCH is provided automatically by BuildKit (amd64 / arm64).
ARG TARGETARCH

# Cap glibc's per-thread malloc arenas. NumPy and TensorFlow are heavily
# multi-threaded, and glibc otherwise creates up to 8 arenas per CPU core, which
# fragments the heap and inflated the container's resident memory by several
# hundred MB on many-core hosts. Two arenas keeps peak RAM well down with no
# measurable speed cost for this workload.
ENV MALLOC_ARENA_MAX=2

# ffmpeg + ffprobe: BtbN static GPL builds, selected per arch. ffprobe reads the
# input's resolution/format; ffmpeg decodes frames. libgomp1 is the OpenMP
# runtime needed by the TensorFlow CPU wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget xz-utils ca-certificates p7zip-full libgomp1 && \
    case "${TARGETARCH}" in \
        amd64) FF_ARCH=linux64 ;; \
        arm64) FF_ARCH=linuxarm64 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    wget -q -O /tmp/ffmpeg.tar.xz \
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-${FF_ARCH}-gpl.tar.xz" && \
    mkdir -p /opt/ffmpeg && \
    tar xf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 && \
    ln -s /opt/ffmpeg/bin/ffmpeg  /usr/local/bin/ffmpeg && \
    ln -s /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe && \
    rm /tmp/ffmpeg.tar.xz && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /resources/
COPY ./requirements.txt /resources/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /resources/requirements.txt

COPY *.py /resources/
COPY ./model_coefficients /resources/model_coefficients

# The image model ships as a split 7-Zip archive; unpack it to
# image_model.h5 (helperfunctions.py loads it at import time) and drop
# the archive parts to keep the image small.
RUN cd /resources/model_coefficients && \
    7z x image_model.h5.7z.001 && \
    rm -f image_model.h5.7z.*

WORKDIR /resources/
