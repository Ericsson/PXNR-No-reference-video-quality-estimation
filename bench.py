"""
Benchmark / memory-profiling harness for PXNR.

Runs the pixel pipeline (YUV decode -> RGB -> quadification -> image model ->
video model) on a given file and reports the wall-clock time, the final MOS and
the peak resident memory. It samples RSS in a background thread (catches short
transient spikes) and also reports the OS-reported maximum (getrusage).

Parameters come from the environment so the same harness can be pointed at the
original or the optimised code without edits:

    PXNR_VIDEO       path to the input video (default: BigBuckBunny clip)
    PXNR_WIDTH       processing width  (default: native width via ffprobe)
    PXNR_HEIGHT      processing height (default: native height via ffprobe)
    PXNR_MAX_FRAMES  cap on extracted frames for fast iteration (default: 96)
    PXNR_FPS         frames per second (default: parsed from filename / 30)
    PXNR_DURATION    duration in seconds used for frame selection (default: 5)
"""

import os
import sys
import time
import threading
import resource
import subprocess

import numpy as np


def ffprobe_dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


class PeakSampler(threading.Thread):
    """Samples this process's RSS every interval seconds; records the peak."""

    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stopev = threading.Event()
        import psutil
        self._proc = psutil.Process()

    def run(self):
        while not self._stopev.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak:
                    self.peak = rss
            except Exception:
                pass
            self._stopev.wait(self.interval)

    def stop(self):
        self._stopev.set()
        self.join(timeout=1.0)


def main():
    video = os.environ.get("PXNR_VIDEO", "./BigBuckBunny_8s_clip.mp4")
    max_frames = int(os.environ.get("PXNR_MAX_FRAMES", "96"))
    duration = float(os.environ.get("PXNR_DURATION", "5"))

    # framerate: filename _NNhz wins, else env, else 30
    import re
    m = re.search(r"_(\d+)hz", os.path.basename(video))
    fps = int(m.group(1)) if m else int(os.environ.get("PXNR_FPS", "30"))

    nat_w, nat_h = ffprobe_dims(video)
    width = int(os.environ.get("PXNR_WIDTH", nat_w))
    height = int(os.environ.get("PXNR_HEIGHT", nat_h))

    # pixel format / depth from the file's first stream
    pix_fmt = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "default=nw=1:nk=1", video],
        capture_output=True, text=True).stdout.strip()
    depth = 10 if "10le" in pix_fmt else 8
    short_pixfmt = pix_fmt.split("p")[0][-3:]

    print(f"bench: video={video} proc={width}x{height} fps={fps} "
          f"dur={duration} max_frames={max_frames} pix_fmt={pix_fmt}", flush=True)

    sampler = PeakSampler()
    sampler.start()

    # Import after sampler starts so model-load memory is captured too.
    import tensorflow as tf
    from helperfunctions import frame_extraction
    from videosources import YUVSourceFFmpeg

    t0 = time.time()
    vsource = YUVSourceFFmpeg(video, width, height, fps, depth, short_pixfmt, False)
    frame_predictions = frame_extraction(vsource, fps, duration, depth,
                                          short_pixfmt, max_frames)

    # The video (aggregation) model has a fixed 96-frame input. Only run it when
    # we extracted the full set; for fast low-frame memory probes, skip it (it
    # is tiny and not where the memory goes).
    if max_frames == 96:
        video_model = tf.keras.models.load_model("./model_coefficients/video_model.h5",
                                                 compile=False)
        final_pred = video_model.predict(frame_predictions, verbose=0)
        mos = float(np.ravel(final_pred)[0])
    else:
        mos = float("nan")
    elapsed = time.time() - t0

    sampler.stop()
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    print(f"RESULT mos={mos:.6f} "
          f"time={elapsed:.1f}s "
          f"peak_sampled_MB={sampler.peak/1e6:.0f} "
          f"peak_maxrss_MB={maxrss/1e6:.0f}", flush=True)


if __name__ == "__main__":
    main()
