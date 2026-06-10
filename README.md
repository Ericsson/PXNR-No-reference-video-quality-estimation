# PXNR-No-reference-video-quality-estimation
The code herein is a sample implementation of the Ericsson submission to the ITU-T SG12 P.NATS phase 2 competition for a no-reference pixel based video quality estimation module.

# License

See the ./licenses/ directory for individual license files for this software together with used and referenced libraries and modules.

# Usage

First, install the prerequisites. Python3, FFmpeg and the necessary libraries and modules are listed below.

Uncompress the 7zip-archive img_model-ep01-vloss40.883.hdf5.7z.001 and .002 of the model coefficients located in the model_coefficients-directory. The resulting file, img_model-ep01-vloss40.883.hdf5, should be located next to the 7z-archive for the code to work.

After you have verified that everything has been installed properly, open a command line terminal on your platform and browse to the directory containing the
code. The model will run on the supplied .mp4-file and will produce an output that looks like this, together with some warnings depending on your platform etc:


```
$ python3 PXNR-model.py

2019-11-19 11:26:21.772936: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
YUVSourceFFmpeg(): filename='./BigBuckBunny_8s_clip.mp4', width=3840, height=2160, pix_fmt=yuv422p, depth=8, fps=30
YUVSourceFFmpeg(): ffmpeg -i ./BigBuckBunny_8s_clip.mp4 -filter:v scale=w=3840:h=2160,fps=30 -f image2pipe -c:v rawvideo -pix_fmt yuv422p -
/lib/python3.5/site-packages/skimage/transform/_warps.py:110: UserWarning: Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
  warn("Anti-aliasing will be enabled by default in skimage 0.15 to "
1/1 [==============================] - 0s
Final prediction of ./BigBuckBunny_8s_clip.mp4 is
MOS: [[2.515198]]
```

If you want to run it on another video file, add the path and proper sample variables in the `PXNR-model.py` file.

# Docker

A containerised version is provided that bundles Python, TensorFlow, FFmpeg and the model coefficients, so you do not have to install any of the prerequisites listed below by hand. The image is multi-platform and builds natively on both Intel/AMD (`linux/amd64`) and ARM, including Apple Silicon (`linux/arm64`).

The official `tensorflow/tensorflow` image is only published for amd64, so the container is instead based on the multi-architecture `python:3.12-slim` image, with TensorFlow installed from PyPI. FFmpeg comes from the static GPL builds by BtbN; the correct build is selected automatically for the target architecture.

### Building

A small helper script, `build.sh`, wraps the Docker BuildKit commands.

To build for the architecture of your own machine and load the image into your local Docker so you can run it right away, run:

```
./build.sh
```

This produces an image tagged `pxnr:latest`.

To build a combined amd64 and arm64 image and push it to a registry, pass `push` and an image reference. A multi-architecture image cannot be loaded into the local Docker store, so it has to be pushed:

```
./build.sh push ghcr.io/your-org/pxnr:latest
```

The first run creates a `pxnr-multiarch` BuildKit builder that is reused afterwards.

If you only need a single-architecture image for your own machine, you can also build it directly without the script:

```
docker build -t pxnr .
```

### Running

The image contains the code and the model coefficients but no test video, so mount the video you want to evaluate into the container and pass its path to `run_pxnr_on_file.py`:

```
docker run --rm -v "$PWD/BigBuckBunny_8s_clip.mp4:/data/in.mp4" pxnr \
  python3 run_pxnr_on_file.py /data/in.mp4
```

The predicted MOS is printed to the console (and also written to a text file in the working directory inside the container). As with the native version, inference runs on the CPU and takes several minutes even for a short clip. An image running under emulation, for example an amd64 image on an ARM host, will be slower still, so prefer a natively built image for real work.

### GPU acceleration (NVIDIA)

The image model is a deep ResNet that runs ~8000 forward passes per scored clip
and dominates the runtime, so moving it to an NVIDIA GPU is the main speed-up.

**Expected benefit.** Measured on a modest GTX 1050 Ti (4 GB), evaluating the
included 8 s clip (96 frames at 4K):

| | CPU (8 cores) | GPU (GTX 1050 Ti) |
| --- | --- | --- |
| image-model inference | ~450 s | ~165 s (≈3x faster) |
| full run (decode → MOS) | ~11 min | ~6.6 min |

The gain scales with the GPU: a larger card will cut inference further, since the
whole 85-sub-frame batch fits in a few hundred MB of VRAM. The remaining time is
mostly the anti-aliased resize on the CPU (set `PXNR_FAST_RESIZE=area` to trade a
small score change for a much faster resize — see "Resize" below).

**What was tested.** The output is unchanged by the GPU: the GPU and CPU image
model agree to ~1e-5 on the interim predictions, and the end-to-end MOS for the
included clip is identical to five decimals (2.558981). Verified natively
(Python 3.12 + `tensorflow[and-cuda]` 2.18) and inside the Docker GPU image
(`--gpus all`), both on the GTX 1050 Ti with driver 550 / CUDA 12.4.

**Native.** Install the GPU dependencies instead of the CPU ones — these bundle
the matching CUDA/cuDNN libraries as pip wheels, so only a recent NVIDIA driver
is needed on the host (no system CUDA toolkit):

```
pip install -r requirements-gpu.txt
python3 PXNR-model.py        # TensorFlow picks up the GPU automatically
```

The code detects the GPU and, since the whole 85-subframe frame fits in a few
hundred MB of VRAM, defaults to pushing it through in a single batch. Override
with `PXNR_IMG_BATCH=<n>` if you are VRAM-constrained.

**Docker.** The GPU image is a separate, amd64-only build (the CUDA wheels have
no arm64 / Apple-Silicon counterpart). It requires the host to have the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed so that `--gpus all` injects the driver into the container. On
Debian/Ubuntu:

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.

Then build and run the GPU image:

```
./build.sh gpu
docker run --rm --gpus all -v "$PWD/BigBuckBunny_8s_clip.mp4:/data/in.mp4" pxnr:gpu \
  python3 run_pxnr_on_file.py /data/in.mp4
```

# Prerequisites

This code has been run, as is, on Mac OS and Ubuntu, producing identical results. FFmpeg, Python and associated libraries have not been included here in this repository and must be installed separately. The model itself can work with other decoders as long as the pictures from the video can be read as arrays using yuv or rgb pixel formats. The user has to modify the code on their own if the use case is anything other than what has been shown in the included example.

### Python

Python3 has been used when developing and running the models herein. The version supplied with the model in the competition was 3.5, but this has been tested with 3.6, 3.8 and 3.12 giving the same results. The updated tensorflow 2 version has only been tested with 3.8+.

These are the modules and versions that were used when running the code originally.

* scikit-image==0.18.3
* tensorflow==2.6.0
* numpy==1.19.5
* h5py==3.1.0

### FFmpeg

This code has been tested with FFmpeg version 3.2.2, 4.1.1, 4.4, and 8.1. The FFmpeg executeable must be included in the path variable for this code to run. Try running `ffmpeg -h` in a terminal, on your device, to see if you have the application in your path.

### Big Buck Bunny clip

The included and heavily encoded video file is an 8s clip taken from the Big Buck Bunny movie, created by the Blender organization. The source material can be found on www.blender.org.

# Upgrade

This repository was upgraded from tensorflow 1.4.1 to 2.6.0. To verify that this did not change the behavior of the model, a number of tests and comparisons was performed. No difference was found on the MOS level and only negligible differences where found on the interim predictions. See a more detailed report in the "Upgrade report"-directory.

### Resize

The default resize is unchanged from the master branch: scikit-image's
anti-aliased `transform.resize`. The anti-aliasing algorithm is numerically
stable across scikit-image versions (the pinned 0.18.3 and a current 0.26 agree
to a mean of 4e-5 on real frames), so the MOS matches the original master
output. The only optimisation in the default path is skipping the sub-frames
that are already at the target size: at 4K input the 64 deepest quad slices are
exactly 270x480, where an anti-aliased resize is the exact identity, so they are
returned untouched.

For users who want maximum speed and can tolerate a small score change, setting
`PXNR_FAST_RESIZE=area` switches the down-sampling to OpenCV's `INTER_AREA`,
which is roughly 90x faster than the anti-aliased skimage resize but only
approximates it and shifts the MOS by about 0.07 on the 1-5 scale. It is
off by default.
