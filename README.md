# PXNR-No-reference-video-quality-estimation

The code herein is a sample implementation of the Ericsson submission to the ITU-T SG12 P.NATS phase 2 competition for a no-reference pixel based video quality estimation module.

The model is also described in [this ITU-T Technical Report](https://www.itu.int/en/publications/ITU-T/Pages/publications.aspx?parent=T-TUT-OPAVQ-2019-PSNTR&media=electronic).

# License

See the `./licenses/` directory for individual license files for this software together with used and referenced libraries and modules.

# Requirements

To run the model natively you need a Python environment, FFmpeg, the Python libraries listed below, and the (compressed) model coefficients unpacked. If you would rather not install any of this by hand, skip to the [Docker](#docker) section, which bundles everything into a container.

### Python

Python **3.9** up to **3.12** (64-bit).

The pinned TensorFlow 2.18 only publishes wheels for that range; 3.12 is what the current code is tested on. The original competition code ran on 3.5 and has also been run on 3.6 and 3.8.

### FFmpeg

`ffmpeg` and `ffprobe` must be installed and on your `PATH` — they decode and probe the input video. Check with `ffmpeg -version`. The code has been used with FFmpeg 3.2.2, 4.1.1, 4.4 and 8.1; any reasonably recent build works.

You can download a build [from here](https://github.com/btbn/ffmpeg-builds/).

### Python libraries

Install them with pip (a virtual environment is recommended):

```
pip install -r requirements.txt
```

This pulls in TensorFlow 2.18, scikit-image (the default anti-aliased resize) and OpenCV (only used by the optional `PXNR_FAST_RESIZE=area` fast path). NumPy and h5py come in transitively via TensorFlow.

To run the image model on an **NVIDIA GPU**, install `requirements-gpu.txt` instead — see [GPU acceleration](#gpu-acceleration-nvidia) for details and the expected speed-up.

### Model coefficients

The image-model weights are checked in as a split 7-Zip archive to keep the file sizes manageable. Before the first run, unpack them so that `image_model.h5` sits next to the archive parts in `model_coefficients/`:

```
cd model_coefficients
7z x image_model.h5.7z.001      # reads .001 and .002, writes image_model.h5
cd ..
```

`7z` comes from the `p7zip-full` package (Debian/Ubuntu: `sudo apt install p7zip-full`; macOS: `brew install p7zip`). The video (aggregation) model, `video_model.h5`, is small and already uncompressed, so it needs no extraction.

### GPU

If you have an NVIDIA GPU and want to use it, see the [GPU acceleration](#gpu-acceleration-nvidia) section for the expected speed-up and installation instructions.

# Usage

Once the requirements above are in place, run the model from the directory containing the code, passing the video file to score as the first argument. The predicted MOS is printed as JSON on standard output; all progress and diagnostic messages go to standard error, so the two streams can be redirected separately:

```bash
python3 PXNR-model.py BigBuckBunny_8s_clip.mp4 --fps 30
```

This produces:

```
YUVSourceFFmpeg(): filename='.../BigBuckBunny_8s_clip.mp4', width=3840, height=2160, pix_fmt=yuv422p, depth=8, fps=30.0
YUVSourceFFmpeg(): ffmpeg -i .../BigBuckBunny_8s_clip.mp4 -filter:v scale=w=3840:h=2160,fps=30.0 -f image2pipe -c:v rawvideo -pix_fmt yuv422p -
...
MOS: 2.558981
{"input": "BigBuckBunny_8s_clip.mp4", "fps": 30.0, "duration": 8.0, "mos": 2.558981}
```

The `YUVSourceFFmpeg(...)`/`MOS:` lines and any TensorFlow/CPU information and warnings go to standard error and are harmless; only the final JSON line is written to standard output.

You can redirect the output to a file if you want to save it:

```bash
PXNR-model.py BigBuckBunny_8s_clip.mp4 --fps 30 > output.json
```

Scoring a short clip takes several minutes on a CPU (see [GPU acceleration](#gpu-acceleration-nvidia) to speed this up).

By default the frame rate is read from a `_NNhz` tag in the filename (falling back to 60); override it with `--fps`, as above. `--duration` and `--pix-fmt` control the analysed length and the decode pixel format.

Run `python3 PXNR-model.py --help` for the full help.

# Docker Usage

A containerised version is provided that bundles Python, TensorFlow, FFmpeg and the model coefficients, so you do not have to install any of the [requirements](#requirements) by hand. The image is multi-platform and builds natively on both Intel/AMD (`linux/amd64`) and ARM, including Apple Silicon (`linux/arm64`).

The official `tensorflow/tensorflow` image is only published for amd64, so the container is instead based on the multi-architecture `python:3.12-slim` image, with TensorFlow installed from PyPI.

FFmpeg comes from the static GPL builds by BtbN; the correct build is selected automatically for the target architecture.

### Building

A small helper script, `build.sh`, wraps the Docker BuildKit commands.

To build for the architecture of your own machine and load the image into your local Docker so you can run it right away, run:

```bash
./build.sh
```

This produces an image tagged `pxnr:latest`. The first run creates a `pxnr-multiarch` BuildKit builder that is reused afterwards.

You can also build it directly without the helper script:

```bash
docker build -t pxnr .
```

### Running

The image's entry point is the model itself, so mount the video you want to evaluate into the container and pass its path as the argument:

```bash
docker run --rm -v "$PWD/BigBuckBunny_8s_clip.mp4:/data/in.mp4" pxnr /data/in.mp4 --fps 30
```

The predicted MOS is printed as JSON on standard output (`{"input": ..., "mos": ...}`); all progress and diagnostics go to standard error. As with the native version, inference runs on the CPU and takes several minutes even for a short clip. An image running under emulation, for example an amd64 image on an ARM host, will be slower still, so prefer a natively built image for real work.

# GPU acceleration (NVIDIA)

The image model is a deep ResNet that runs ~8000 forward passes per scored clip and dominates the runtime, so moving it to an NVIDIA GPU is the main speed-up.

## Native Usage

Install the GPU dependencies instead of the CPU ones — these bundle the matching CUDA/cuDNN libraries as pip wheels, so only a recent NVIDIA driver is needed on the host (no system CUDA toolkit):

```bash
pip install -r requirements-gpu.txt
python3 PXNR-model.py        # TensorFlow picks up the GPU automatically
```

The code detects the GPU and, since the whole 85-subframe frame fits in a few hundred MB of VRAM, defaults to pushing it through in a single batch. Override with `PXNR_IMG_BATCH=<n>` if you are VRAM-constrained.

## Docker Usage

The GPU image is a separate, amd64-only build (the CUDA wheels have no arm64 / Apple-Silicon counterpart). It requires the host to have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed so that `--gpus all` injects the driver into the container. On Debian/Ubuntu:

```bash
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

```bash
./build.sh gpu
docker run --rm --gpus all -v "$PWD/BigBuckBunny_8s_clip.mp4:/data/in.mp4" pxnr:gpu /data/in.mp4 --fps 30
```

## Expected Speedup

Measured on a modest GTX 1050 Ti (4 GB), evaluating the included 8 s clip (96 frames at 4K):

|                         | CPU (8 cores) | GPU (GTX 1050 Ti)   |
| ----------------------- | ------------- | ------------------- |
| image-model inference   | ~450 s        | ~165 s (≈3x faster) |
| full run (decode → MOS) | ~11 min       | ~6.6 min            |

The gain scales with the GPU: a larger card will cut inference further, since the whole 85-sub-frame batch fits in a few hundred MB of VRAM. The remaining time is mostly the anti-aliased resize on the CPU (set `PXNR_FAST_RESIZE=area` to trade a small score change for a much faster resize — see "Resize" below).

The output is unchanged by the GPU – the GPU and CPU image model agree to ~1e-5 on the interim predictions, and the end-to-end MOS for the included clip is identical to five decimals (2.558981). Verified natively (Python 3.12 + `tensorflow[and-cuda]` 2.18) and inside the Docker GPU image (`--gpus all`), both on the GTX 1050 Ti with driver 550 / CUDA 12.4.

# Further Notes

This code has been run, as is, on macOS and Ubuntu, producing identical results. The model can work with other decoders as long as the frames can be read as arrays in a YUV or RGB pixel format; the user has to adapt the code for any use case other than the included example.

The bundled `BigBuckBunny_8s_clip.mp4` is a heavily encoded 8 s clip from the Big Buck Bunny movie by the Blender organization. The source material can be found at www.blender.org.

## Upgrade

This repository was upgraded from TensorFlow 1.4.1 to 2.6.0, and later to 2.18. To verify that this did not change the behavior of the model, a number of tests and comparisons were performed. No difference was found on the MOS level and only negligible differences on the interim predictions. See the detailed report in the "Upgrade report" directory. The module versions used when the code originally ran were `tensorflow==2.6.0`, `scikit-image==0.18.3`, `numpy==1.19.5` and `h5py==3.1.0`; the current pinned versions are in `requirements.txt`.

## Resize

The default resize is unchanged from the master branch: scikit-image's anti-aliased `transform.resize`. The anti-aliasing algorithm is numerically stable across scikit-image versions (the pinned 0.18.3 and a current 0.26 agree to a mean of 4e-5 on real frames), so the MOS matches the original master output. The only optimisation in the default path is skipping the sub-frames that are already at the target size: at 4K input the 64 deepest quad slices are exactly 270x480, where an anti-aliased resize is the exact identity, so they are returned untouched.

For users who want maximum speed and can tolerate a small score change, setting `PXNR_FAST_RESIZE=area` switches the down-sampling to OpenCV's `INTER_AREA`, which is roughly 90x faster than the anti-aliased skimage resize but only approximates it and shifts the MOS by about 0.07 on the 1-5 scale. It is off by default.
